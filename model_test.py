import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ckpt_dir = r"D:/projects/llm-finetune/tinyLLama"  # from trainer_state.json

# Load base model (4-bit if you want, same as training)
from transformers import BitsAndBytesConfig
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

# Attach LoRA weights
model = PeftModel.from_pretrained(model, ckpt_dir)

tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)  # uses tokenizer.json, tokenizer_config, special_tokens_map
model.eval()

# Test the model with a sample prompt related to CUAD (contract QA)
messages = [
    {"role": "user", "content": "Based on the following contract excerpt: 'The agreement is between Company A and Company B. The term is 5 years.' What is the term of the agreement?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {messages[0]['content']}")
print(f"Generated: {generated_text[len(prompt):].strip()}")  # Only show the new part
