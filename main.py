import functools
import gc
import os
from typing import List, Optional

import einops
import gradio as gr
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

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

    def get_instructions(file_path, column_name):
        dataset = pd.read_csv(file_path)
        instructions = dataset[column_name].tolist()
        train, test = train_test_split(instructions, test_size=0.2, random_state=42)
        return train, test

    def tokenize_instructions(tokenizer, instructions):
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True
        ) for instruction in instructions]
        return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

    def get_orthogonalized_matrix(matrix, vec):
        proj = einops.einsum(matrix, vec.view(-1, 1), '... d_model, d_model single -> ... single') * vec
        return matrix - proj

    try:
        p(0, "正在加载模型...")

        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            n_devices=1,
            dtype=torch.bfloat16,
            default_padding_side='left'
        )
        model.tokenizer.padding_side = 'left'
        model.tokenizer.pad_token = '<|extra_0|>'

        tokenize_instructions_fn = functools.partial(tokenize_instructions, tokenizer=model.tokenizer)
        model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        p(0.1, "正在加载有害数据库...")
        harmful_inst_train, harmful_inst_test = get_instructions("harmful_behaviors.csv", 'goal')

        p(0.15, "正在加载无害数据库...")
        harmless_inst_train, harmless_inst_test = get_instructions("./harmless_behaviors", 'instruction')

        p(0.2, "正在分词有害数据库...")
        harmful_tokens = tokenize_instructions_fn(instructions=harmful_inst_train[:n_inst_train])

        p(0.25, "正在分词无害数据库...")
        harmless_tokens = tokenize_instructions_fn(instructions=harmless_inst_train[:n_inst_train])

        p(0.3, "正在获取有害数据库中间过程参数...")
        harmful_logits, harmful_cache = model.run_with_cache(harmful_tokens,
                                                             names_filter=lambda hook_name: 'resid' in hook_name)
        harmful_logits.cpu()
        torch.cuda.empty_cache()

        p(0.5, "正在获取无害数据库中间过程参数...")
        harmless_logits, harmless_cache = model.run_with_cache(harmless_tokens,
                                                               names_filter=lambda hook_name: 'resid' in hook_name)
        harmless_logits.cpu()
        torch.cuda.empty_cache()

        p(0.7, "正在计算有害与无害激活平均差异...")
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
        weights = torch.randn(len(filtered_tensors))
        weights = torch.softmax(weights, dim=0)
        combined_tensor = torch.sum(torch.stack([w * t for w, t in zip(weights, filtered_tensors)]), dim=0)

        p(0.72, "正在移除原模型安全方向向量...")

        print(f"模型总层数：{len(model.blocks)} 当前提取用层数：{layer}")

        blocks: List[List[Optional[torch.Tensor]]] = [[None, None] for _ in range(len(model.blocks))]

        for i in range(len(model.blocks)):
            blocks[i][0] = get_orthogonalized_matrix(model.blocks[i].attn.W_O, combined_tensor)
            blocks[i][1] = get_orthogonalized_matrix(model.blocks[i].mlp.W_out, combined_tensor)

        model_W_E_data = get_orthogonalized_matrix(model.W_E, combined_tensor)

        p(0.73, "正在清理内存与显存...")
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        p(0.75, "正在创建原模型副本...")
        save_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)

        p(0.95, "正在写入新向量到新模型...")
        with torch.no_grad():
            save_model.model.embed_tokens.weight.data = model_W_E_data.contiguous()
            size = blocks[layer[0]][0].shape[-1]
            for i in range(len(blocks)):
                save_model.model.layers[i].self_attn.o_proj.weight.data = blocks[i][0].view(-1, size).T.contiguous()
                save_model.model.layers[i].mlp.down_proj.weight.data = blocks[i][1].T.contiguous()
        save_model.save_pretrained(f"{model_name}-Without-Refusal")
        AutoTokenizer.from_pretrained(model_name).save_pretrained(f"{model_name}-Without-Refusal")

        p(1, "完成")
        save_model = save_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        if 'save_model' in locals():
            del save_model

        return ["🤗 创建成功\n\n已保存模型到" + os.getcwd() + f"/{model_name}-Without-Refusal", "已清理显卡显存占用"]
    except Exception as e:
        print(e)

        # 检查并删除变量
        for var_name in ['model', 'save_model']:
            if var_name in locals():
                del locals()[var_name]

        # 清理显存
        torch.cuda.empty_cache()

        return ["🤯 发生错误\n\n" + str(e), "已清理显卡显存占用"]


# 无界面运行
# process("Qwen/Qwen2-7B-Instruct", 19, 1, "12", "cuda")

app = gr.Interface(fn=process, inputs=[
    gr.Dropdown(
        OFFICIAL_MODEL_NAMES,
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

app.launch(share=False, server_port=6001, server_name="0.0.0.0")
