import functools
import gc
import os
from typing import List, Optional

import einops
import gradio as gr
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_instructions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        instructions = file.readlines()

    instructions = [line.strip() for line in instructions]
    return instructions


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


def get_diverted_matrix(matrix, vec):
    # 计算每一行与向量 v 的点积
    dot_products = torch.matmul(matrix, vec)

    # 将点积结果扩展为与 A 相同的形状
    dot_products_expanded = dot_products.unsqueeze(1).expand_as(matrix)

    # 将扩展后的点积结果与原矩阵 A 相加
    A_adjusted = matrix + dot_products_expanded

    return A_adjusted


def remove_mode(model_name, n_inst_train, refusal_dir_coefficient, layer, device, harmful_behaviors_file_path,
                harmless_behaviors_file_path, progress=gr.Progress()):
    def p(percentage, message):
        print(f"{percentage * 100}%: {message}")
        progress(percentage, message)

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
        harmful_inst_train = get_instructions(harmful_behaviors_file_path)

        p(0.15, "正在加载无害数据库...")
        harmless_inst_train = get_instructions(harmless_behaviors_file_path)

        p(0.2, "正在分词有害数据库...")
        harmful_tokens = tokenize_instructions_fn(instructions=harmful_inst_train[:n_inst_train])

        p(0.25, "正在分词无害数据库...")
        harmless_tokens = tokenize_instructions_fn(instructions=harmless_inst_train[:n_inst_train])

        p(0.3, "正在获取有害数据库中间过程参数...")
        harmful_logits, harmful_cache = model.run_with_cache(harmful_tokens,
                                                             names_filter=lambda hook_name: 'resid' in hook_name)
        harmful_logits = harmful_logits.cpu()
        del harmful_logits
        clear_gpu_cache()

        p(0.5, "正在获取无害数据库中间过程参数...")
        harmless_logits, harmless_cache = model.run_with_cache(harmless_tokens,
                                                               names_filter=lambda hook_name: 'resid' in hook_name)
        harmless_logits = harmless_logits.cpu()
        del harmless_logits
        clear_gpu_cache()

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

        del harmful_cache, harmless_cache
        torch.cuda.empty_cache()

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
        model = model.cpu()
        del model
        clear_gpu_cache()

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
        clear_gpu_cache()

        if 'save_model' in locals():
            del save_model

        print("🤗 创建成功\n\n已保存模型到" + os.getcwd() + f"/{model_name}-Without-Refusal")
        return "🤗 创建成功\n\n已保存模型到" + os.getcwd() + f"/{model_name}-Without-Refusal"
    except Exception as e:
        print(e)

        # 检查并删除变量
        for var_name in ['model', 'save_model']:
            if var_name in locals():
                del locals()[var_name]

        # 清理显存
        clear_gpu_cache()

        return ["🤯 发生错误\n\n" + str(e), "已清理显卡显存占用"]


def divert_mode(model_name, n_inst_train, target_dir_coefficient, layer, device, target_prompt,
                basic_behaviors_file_path, progress=gr.Progress()):
    def p(percentage, message):
        print(f"{percentage * 100}%: {message}")
        progress(percentage, message)

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

        p(0.1, "正在加载普通Prompt数据库...")
        from_inst_train = get_instructions(basic_behaviors_file_path)

        to_inst_train = [line + target_prompt for line
                         in
                         from_inst_train]

        p(0.2, "正在分词普通数据库...")
        from_tokens = tokenize_instructions_fn(instructions=from_inst_train[:n_inst_train])

        p(0.25, "正在分词目标数据库...")
        to_tokens = tokenize_instructions_fn(instructions=to_inst_train[:n_inst_train])

        p(0.3, "正在获取普通数据库中间过程参数...")
        from_logits, from_cache = model.run_with_cache(from_tokens,
                                                       names_filter=lambda hook_name: 'resid' in hook_name)
        from_logits = from_logits.cpu()
        del from_logits
        clear_gpu_cache()

        p(0.5, "正在获取目标数据库中间过程参数...")
        to_logits, to_cache = model.run_with_cache(to_tokens,
                                                   names_filter=lambda hook_name: 'resid' in hook_name)
        to_logits = to_logits.cpu()
        del to_logits
        clear_gpu_cache()

        p(0.7, "正在计算普通与目标激活平均差异...")
        pos = -1

        target_dir: List[Optional[torch.Tensor]] = [None for _ in range(len(model.blocks))]

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
            from_mean_act = from_cache['resid_pre', i][:, pos, :].mean(dim=0)
            to_mean_act = to_cache['resid_pre', i][:, pos, :].mean(dim=0)
            target_dir[i] = to_mean_act - from_mean_act
            target_dir[i] = target_dir[i] / target_dir[i].norm()
            target_dir[i] = target_dir[i] * target_dir_coefficient

        del from_mean_act, to_mean_act
        torch.cuda.empty_cache()

        filtered_tensors = [t for t in target_dir if t is not None]
        weights = torch.randn(len(filtered_tensors))
        weights = torch.softmax(weights, dim=0)
        combined_tensor = torch.sum(torch.stack([w * t for w, t in zip(weights, filtered_tensors)]), dim=0)

        p(0.72, "正在转向向量...")

        print(f"模型总层数：{len(model.blocks)} 当前提取用层数：{layer}")

        blocks: List[List[Optional[torch.Tensor]]] = [[None, None] for _ in range(len(model.blocks))]

        for i in range(len(model.blocks)):
            blocks[i][0] = get_diverted_matrix(model.blocks[i].attn.W_O, combined_tensor)
            blocks[i][1] = get_diverted_matrix(model.blocks[i].mlp.W_out, combined_tensor)

        model_W_E_data = get_diverted_matrix(model.W_E, combined_tensor)

        p(0.73, "正在清理内存与显存...")
        model = model.cpu()
        del model
        clear_gpu_cache()

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
        clear_gpu_cache()

        if 'save_model' in locals():
            del save_model

        print("🤗 创建成功\n\n已保存模型到" + os.getcwd() + f"/{model_name}-Diverted")
        return "🤗 创建成功\n\n已保存模型到" + os.getcwd() + f"/{model_name}-Diverted"
    except Exception as e:
        print(e)

        # 检查并删除变量
        for var_name in ['model', 'save_model']:
            if var_name in locals():
                del locals()[var_name]

        # 清理显存
        clear_gpu_cache()

        return ["🤯 发生错误\n\n" + str(e), "已清理显卡显存占用"]
