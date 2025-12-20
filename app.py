import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr

# ---- Load model + LoRA (same as your working script) ----
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ckpt_dir = r"D:/projects/llm-finetune/tinyLLama"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ckpt_dir)
tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
model.eval()

# ---- Generation function ----
def answer(question, context, max_new_tokens=64, temperature=0.7, top_p=0.9):
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full[len(prompt):].strip()

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("# TinyLlama CUAD QA")
    with gr.Row():
        question = gr.Textbox(label="Question")
    context = gr.Textbox(label="Contract context", lines=8)
    with gr.Row():
        max_new = gr.Slider(16, 256, value=64, step=8, label="Max new tokens")
        temp = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
    output = gr.Textbox(label="Model answer", lines=4)

    run_btn = gr.Button("Generate answer")
    run_btn.click(
        answer,
        inputs=[question, context, max_new, temp, top_p],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
