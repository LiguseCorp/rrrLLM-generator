import functools
import gc
import os
from typing import List, Optional

import einops
import gradio as gr
import pandas as pd
import torch
from datasets import load_from_disk
from jaxtyping import Float, Int
from sklearn.model_selection import train_test_split
from torch import Tensor
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)


def gpu_usage():
    def inner():
        r = os.popen("nvidia-smi")
        text = r.read()
        r.close()
        return text

    return inner


def process(model_name, n_inst_train, refusal_dir_coefficient, layer, device, progress=gr.Progress()):
    def p(percentage, message):
        print(f"{percentage * 100}%: {message}")
        progress(percentage, message)

    try:
        p(0, "正在加载模型...")
        model: HookedTransformer = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            # move_to_device=False,
            n_devices=1,
            dtype=torch.bfloat16,
            # hf_model=model,
            default_padding_side='left'
        )

        model.tokenizer.padding_side = 'left'
        model.tokenizer.pad_token = '<|extra_0|>'

        def get_harmful_instructions():
            # url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
            # response = requests.get(url)
            #
            # dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            dataset = pd.read_csv("harmful_behaviors.csv")
            instructions = dataset['goal'].tolist()

            train, test = train_test_split(instructions, test_size=0.2, random_state=42)
            return train, test

        def get_harmless_instructions():
            # hf_path = 'tatsu-lab/alpaca'
            # dataset = load_dataset(hf_path)

            dataset = load_from_disk("./harmless_behaviors")

            # 筛选输入类型的数据
            instructions = []
            for i in range(len(dataset['train'])):
                if dataset['train'][i]['input'].strip() == '':
                    instructions.append(dataset['train'][i]['instruction'])

            train, test = train_test_split(instructions, test_size=0.2, random_state=42)
            return train, test

        p(0.1, "正在加载有害数据库...")
        harmful_inst_train, harmful_inst_test = get_harmful_instructions()

        p(0.15, "正在加载无害数据库...")
        harmless_inst_train, harmless_inst_test = get_harmless_instructions()

        def tokenize_instructions_chat(
                tokenizer,
                instructions: List[str]
        ) -> Int[Tensor, 'batch_size seq_len']:
            prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True
            ) for instruction in instructions]

            return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

        tokenize_instructions_fn = functools.partial(tokenize_instructions_chat, tokenizer=model.tokenizer)

        model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        p(0.2, "正在分词有害数据库...")
        harmful_tokens = tokenize_instructions_fn(instructions=harmful_inst_train[:n_inst_train])

        p(0.25, "正在分词无害数据库...")
        harmless_tokens = tokenize_instructions_fn(instructions=harmless_inst_train[:n_inst_train])

        p(0.3, "正在获取有害数据库中间过程参数...")
        harmful_logits, harmful_cache = model.run_with_cache(harmful_tokens,
                                                             names_filter=lambda hook_name: 'resid' in hook_name)
        harmful_logits = harmful_logits.cpu()
        torch.cuda.empty_cache()

        p(0.5, "正在获取有害数据库中间过程参数...")
        harmless_logits, harmless_cache = model.run_with_cache(harmless_tokens,
                                                               names_filter=lambda hook_name: 'resid' in hook_name)
        harmless_logits = harmless_logits.cpu()
        torch.cuda.empty_cache()

        p(0.7, "正在计算有害与无害激活平均差异...")

        # 在中间层计算有害和无害激活的平均差异。
        pos = -1

        refusal_dir: List[Optional[torch.Tensor]] = [None for _ in range(len(model.blocks))]

        if layer == "-2":
            layer = [i for i in range(len(model.blocks))]
        elif layer == "-1":
            layer = [int(len(model.blocks) / 2)]
        elif len(layer.split(",")) != 1:
            layer = [int(i) for i in layer.split(",")]
        else:
            layer = [int(layer)]

        if int(len(model.blocks) / 2) not in layer:
            layer.append(int(len(model.blocks) / 2))

        for i in layer:
            harmful_mean_act = harmful_cache['resid_pre', i][:, pos, :].mean(dim=0)
            harmless_mean_act = harmless_cache['resid_pre', i][:, pos, :].mean(dim=0)

            refusal_dir[i] = harmful_mean_act - harmless_mean_act

            refusal_dir[i] = refusal_dir[i] / refusal_dir[i].norm()
            refusal_dir[i] = refusal_dir[i] * refusal_dir_coefficient

        filtered_tensors = [t for t in refusal_dir if t is not None]

        # 计算每个tensor的权重，按照正态分布
        weights = torch.randn(len(filtered_tensors))
        weights = torch.softmax(weights, dim=0)  # 使用softmax将权重归一化

        # 按照权重合成单一的tensor
        combined_tensor = torch.sum(torch.stack([w * t for w, t in zip(weights, filtered_tensors)]), dim=0)

        def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, 'd_model']) -> \
                Float[Tensor, '... d_model']:
            proj = einops.einsum(matrix, vec.view(-1, 1), '... d_model, d_model single -> ... single') * vec
            return matrix - proj

        p(0.72, "正在移除原模型安全方向向量...")

        print(f"模型总层数：{len(model.blocks)} 当前提取用层数：{layer}")

        blocks: List[List[Optional[torch.Tensor]]] = [[None, None] for _ in range(len(model.blocks))]

        for i in range(len(model.blocks)):
            blocks[i][0] = get_orthogonalized_matrix(model.blocks[i].attn.W_O,
                                                     combined_tensor)
            blocks[i][1] = get_orthogonalized_matrix(model.blocks[i].mlp.W_out,
                                                     combined_tensor)

        model_W_E_data = get_orthogonalized_matrix(model.W_E, combined_tensor)

        p(0.73, "正在清理内存与显存...")
        # 清理内存
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        p(0.75, "正在创建原模型副本...")
        save_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        )

        def begin_save_model(blocks):
            with torch.no_grad():
                save_model.model.embed_tokens.weight.data = model_W_E_data.contiguous()

                size = blocks[layer[0]][0].shape[-1]

                for i in range(len(blocks)):
                    save_model.model.layers[i].self_attn.o_proj.weight.data = blocks[i][0].view(-1, size).T.contiguous()
                    save_model.model.layers[i].mlp.down_proj.weight.data = blocks[i][1].T.contiguous()

            save_model.save_pretrained(f"{model_name}-Without-Refusal")
            AutoTokenizer.from_pretrained(model_name).save_pretrained(f"{model_name}-Without-Refusal")

        p(0.95, "正在写入新向量到新模型...")
        begin_save_model(blocks)

        p(1, "完成")

        save_model = save_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        try:
            del save_model
        except UnboundLocalError:
            pass

        return ["🤗 创建成功\n\n已保存模型到" + os.getcwd() + f"/{model_name}-Without-Refusal", "已清理显卡显存占用"]
    except Exception as e:
        print(e)
        try:
            del model
        except UnboundLocalError:
            pass

        try:
            del save_model
        except UnboundLocalError:
            pass

        torch.cuda.empty_cache()
        return ["🤯 发生错误\n\n" + e.__str__(), "已清理显卡显存占用"]


# process("01-ai/Yi-6B-Chat",19,1.0,"-1","cuda")
process("Qwen/Qwen2-7B-Instruct", 19, 1, "12", "cuda")
# process("/extdatas/lora/lora/huggingface/hub_cache/models--Qwen--Qwen2-7B-Instruct",24,1,"-1","cuda")

demo = gr.Interface(fn=process, inputs=[
    gr.Dropdown(
        ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2', 'facebook/opt-125m', 'facebook/opt-1.3b',
         'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b',
         'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B',
         'EleutherAI/gpt-neox-20b', 'stanford-crfm/alias-gpt2-small-x21', 'stanford-crfm/battlestar-gpt2-small-x49',
         'stanford-crfm/caprica-gpt2-small-x81', 'stanford-crfm/darkmatter-gpt2-small-x343',
         'stanford-crfm/expanse-gpt2-small-x777', 'stanford-crfm/arwen-gpt2-medium-x21',
         'stanford-crfm/beren-gpt2-medium-x49', 'stanford-crfm/celebrimbor-gpt2-medium-x81',
         'stanford-crfm/durin-gpt2-medium-x343', 'stanford-crfm/eowyn-gpt2-medium-x777', 'EleutherAI/pythia-14m',
         'EleutherAI/pythia-31m', 'EleutherAI/pythia-70m', 'EleutherAI/pythia-160m', 'EleutherAI/pythia-410m',
         'EleutherAI/pythia-1b', 'EleutherAI/pythia-1.4b', 'EleutherAI/pythia-2.8b', 'EleutherAI/pythia-6.9b',
         'EleutherAI/pythia-12b', 'EleutherAI/pythia-70m-deduped', 'EleutherAI/pythia-160m-deduped',
         'EleutherAI/pythia-410m-deduped', 'EleutherAI/pythia-1b-deduped', 'EleutherAI/pythia-1.4b-deduped',
         'EleutherAI/pythia-2.8b-deduped', 'EleutherAI/pythia-6.9b-deduped', 'EleutherAI/pythia-12b-deduped',
         'EleutherAI/pythia-70m-v0', 'EleutherAI/pythia-160m-v0', 'EleutherAI/pythia-410m-v0',
         'EleutherAI/pythia-1b-v0', 'EleutherAI/pythia-1.4b-v0', 'EleutherAI/pythia-2.8b-v0',
         'EleutherAI/pythia-6.9b-v0', 'EleutherAI/pythia-12b-v0', 'EleutherAI/pythia-70m-deduped-v0',
         'EleutherAI/pythia-160m-deduped-v0', 'EleutherAI/pythia-410m-deduped-v0', 'EleutherAI/pythia-1b-deduped-v0',
         'EleutherAI/pythia-1.4b-deduped-v0', 'EleutherAI/pythia-2.8b-deduped-v0', 'EleutherAI/pythia-6.9b-deduped-v0',
         'EleutherAI/pythia-12b-deduped-v0', 'EleutherAI/pythia-160m-seed1', 'EleutherAI/pythia-160m-seed2',
         'EleutherAI/pythia-160m-seed3', 'NeelNanda/SoLU_1L_v9_old', 'NeelNanda/SoLU_2L_v10_old',
         'NeelNanda/SoLU_4L_v11_old', 'NeelNanda/SoLU_6L_v13_old', 'NeelNanda/SoLU_8L_v21_old',
         'NeelNanda/SoLU_10L_v22_old', 'NeelNanda/SoLU_12L_v23_old', 'NeelNanda/SoLU_1L512W_C4_Code',
         'NeelNanda/SoLU_2L512W_C4_Code', 'NeelNanda/SoLU_3L512W_C4_Code', 'NeelNanda/SoLU_4L512W_C4_Code',
         'NeelNanda/SoLU_6L768W_C4_Code', 'NeelNanda/SoLU_8L1024W_C4_Code', 'NeelNanda/SoLU_10L1280W_C4_Code',
         'NeelNanda/SoLU_12L1536W_C4_Code', 'NeelNanda/GELU_1L512W_C4_Code', 'NeelNanda/GELU_2L512W_C4_Code',
         'NeelNanda/GELU_3L512W_C4_Code', 'NeelNanda/GELU_4L512W_C4_Code', 'NeelNanda/Attn_Only_1L512W_C4_Code',
         'NeelNanda/Attn_Only_2L512W_C4_Code', 'NeelNanda/Attn_Only_3L512W_C4_Code',
         'NeelNanda/Attn_Only_4L512W_C4_Code', 'NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr',
         'NeelNanda/SoLU_1L512W_Wiki_Finetune', 'NeelNanda/SoLU_4L512W_Wiki_Finetune', 'ArthurConmy/redwood_attn_2l',
         'llama-7b-hf', 'llama-13b-hf', 'llama-30b-hf', 'llama-65b-hf', 'meta-llama/Llama-2-7b-hf',
         'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf',
         'meta-llama/Llama-2-70b-chat-hf', 'CodeLlama-7b-hf', 'CodeLlama-7b-Python-hf', 'CodeLlama-7b-Instruct-hf',
         'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-70B',
         'meta-llama/Meta-Llama-3-70B-Instruct', 'Baidicoot/Othello-GPT-Transformer-Lens', 'bert-base-cased',
         'roneneldan/TinyStories-1M', 'roneneldan/TinyStories-3M', 'roneneldan/TinyStories-8M',
         'roneneldan/TinyStories-28M', 'roneneldan/TinyStories-33M', 'roneneldan/TinyStories-Instruct-1M',
         'roneneldan/TinyStories-Instruct-3M', 'roneneldan/TinyStories-Instruct-8M',
         'roneneldan/TinyStories-Instruct-28M', 'roneneldan/TinyStories-Instruct-33M',
         'roneneldan/TinyStories-1Layer-21M', 'roneneldan/TinyStories-2Layers-33M',
         'roneneldan/TinyStories-Instuct-1Layer-21M', 'roneneldan/TinyStories-Instruct-2Layers-33M',
         'stabilityai/stablelm-base-alpha-3b', 'stabilityai/stablelm-base-alpha-7b',
         'stabilityai/stablelm-tuned-alpha-3b', 'stabilityai/stablelm-tuned-alpha-7b', 'mistralai/Mistral-7B-v0.1',
         'mistralai/Mistral-7B-Instruct-v0.1', 'mistralai/Mixtral-8x7B-v0.1', 'mistralai/Mixtral-8x7B-Instruct-v0.1',
         'bigscience/bloom-560m', 'bigscience/bloom-1b1', 'bigscience/bloom-1b7', 'bigscience/bloom-3b',
         'bigscience/bloom-7b1', 'bigcode/santacoder', 'Qwen/Qwen-1_8B', 'Qwen/Qwen-7B', 'Qwen/Qwen-14B',
         'Qwen/Qwen-1_8B-Chat', 'Qwen/Qwen-7B-Chat', 'Qwen/Qwen-14B-Chat', 'Qwen/Qwen1.5-0.5B',
         'Qwen/Qwen1.5-0.5B-Chat', 'Qwen/Qwen1.5-1.8B', 'Qwen/Qwen1.5-1.8B-Chat', 'Qwen/Qwen1.5-4B',
         'Qwen/Qwen1.5-4B-Chat', 'Qwen/Qwen1.5-7B', 'Qwen/Qwen1.5-7B-Chat', 'Qwen/Qwen1.5-14B', 'Qwen/Qwen1.5-14B-Chat',
         'Qwen/Qwen2-0.5B', 'Qwen/Qwen2-0.5B-Instruct', 'Qwen/Qwen2-1.5B', 'Qwen/Qwen2-1.5B-Instruct', 'Qwen/Qwen2-7B',
         'Qwen/Qwen2-7B-Instruct', 'microsoft/phi-1', 'microsoft/phi-1_5', 'microsoft/phi-2',
         'microsoft/Phi-3-mini-4k-instruct', 'google/gemma-2b', 'google/gemma-7b', 'google/gemma-2b-it',
         'google/gemma-7b-it', 'google/gemma-2-2b', 'google/gemma-2-2b-it', 'google/gemma-2-9b', 'google/gemma-2-9b-it',
         'google/gemma-2-27b', 'google/gemma-2-27b-it', '01-ai/Yi-6B', '01-ai/Yi-34B', '01-ai/Yi-6B-Chat',
         '01-ai/Yi-34B-Chat', 'google-t5/t5-small', 'google-t5/t5-base', 'google-t5/t5-large', 'ai-forever/mGPT'],
        label="model_name", info="需处理模型（目前仅支持列表中的模型）", value="Qwen/Qwen2-1.5B-Instruct"
    ),
    gr.Slider(1, 500, value=32, label="n_inst_train",
              info="拒绝方向钓鱼例句数量\n更加精准的判别拒绝方向（越大越准确）（高显存需求）"),
    gr.Slider(0.5, 2, value=1, label="refusal_dir_coefficient",
              info="去除拒绝方向系数（增强去除效果。会显著影响大模型自身能力。）（默认为1）"),
    gr.Text(label="layer", value="-1", info="提取特征层（-1为取模型中间层，-2为对每一层独立进行处理（高显存需求））"),
    gr.Text(label="device", value="cuda", info="运行设备")

], outputs=[gr.Text(label="状态", value="就绪"),
            gr.Textbox(label="显卡实时状态",
                       value=gpu_usage(),
                       every=3)], title="去除安全审查大模型生成器",
                    description="通过对检测到的拒绝方向进行正交移除，实现生成去除安全审查后的大模型。")

demo.launch(share=False, server_port=6001, server_name="0.0.0.0")
