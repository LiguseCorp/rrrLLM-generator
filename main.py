import argparse
import os

import gradio as gr
import torch
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

import core
import platform

default_device = ""
default_ip = ""

if torch.cuda.is_available():
    default_device = "cuda"
elif torch.backends.mps.is_available():
    default_device = "mps"
else:
    default_device = "cpu"

if platform.system() == "Windows":
    default_ip = "127.0.0.1"
else:
    default_ip = "0.0.0.0"

parser = argparse.ArgumentParser(description="rrrLLM")
parser.add_argument('--host', type=str, default=default_ip,
                    help='Host IP the web interface running on')
parser.add_argument('--port', type=int, default=8080, help='Port the web interface running on')
parser.add_argument('--cli', action="store_true", default=False, help='Use CLI instead of web interface')
parser.add_argument('--model-name', type=str, default="Qwen/Qwen2-1.5B-Instruct",
                    help='The model to be processed. Default is Qwen/Qwen2-1.5B-Instruct')
parser.add_argument('--n-inst-train', type=int, default=32,
                    help='The number of training instances for refusal direction. Default is 32')
parser.add_argument('--refusal-dir-coefficient', type=float, default=1,
                    help='The coefficient for refusal direction. Default is 1')
parser.add_argument('--layer', type=str, default="-1", help='The layer to be processed. Default is -1')
parser.add_argument('--device', type=str, default=default_device,
                    help=f'The device to run on. Default is {default_device}')

args, unknown = parser.parse_known_args()

torch.set_grad_enabled(False)


def gpu_usage():
    def inner():
        if torch.cuda.is_available():
            r = os.popen("nvidia-smi")
            text = r.read()
            r.close()
            return text
        else:
            return "仅支持cuda环境"

    return inner


if args.cli:
    print("\n\n\n----------CLI Mode----------\n")
    print(f"""Arguments:
    - Model Name: {args.model_name}
    - Number of Training Instances: {args.n_inst_train}
    - Refusal Direction Coefficient: {args.refusal_dir_coefficient}
    - Layer: {args.layer if args.layer != "-1" else "Middle(-1)"}
    - Device: {args.device}
    """)
    print("Continue? (y/n)")
    if input() == "y":
        core.process(args.model_name, args.n_inst_train, args.refusal_dir_coefficient, args.layer, args.device)
    else:
        print("Use --help for help")
        exit()
else:
    app = gr.Interface(fn=core.process, inputs=[
        gr.Dropdown(
            OFFICIAL_MODEL_NAMES,
            label="model_name", info="需处理模型（目前仅支持列表中的模型）", value="Qwen/Qwen2-1.5B-Instruct"
        ),
        gr.Slider(1, 500, value=32, label="n_inst_train",
                  info="拒绝方向钓鱼例句数量\n更加精准的判别拒绝方向（越大越准确）（高显存需求）"),
        gr.Slider(0.5, 2, value=1, label="refusal_dir_coefficient",
                  info="去除拒绝方向系数（增强去除效果。会显著影响大模型自身能力。）（默认为1）"),
        gr.Text(label="layer", value="-1", info="提取特征层（-1为取模型中间层，-2为对每一层独立进行处理（高显存需求））"),
        gr.Text(label="device", value=default_device, info="运行设备")

    ], outputs=[gr.Text(label="状态", value="就绪"),
                gr.Textbox(label="显卡实时状态",
                           value=gpu_usage(),
                           every=3)], title="rrrLLM Generator",
                       description="通过对层中检测到的拒绝方向进行消除，实现生成更低拒绝回答率的大模型。")

    app.launch(share=False, server_port=args.port, server_name=args.host)
