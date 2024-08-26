import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct-Without-Refusal"
# model_name = "01-ai/Yi-6B-Chat-Without-Refusal"

DEVICE = 'cuda'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(model_name)

while True:
    prompt = input("> ")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=128,
        temperature=0.01
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)
