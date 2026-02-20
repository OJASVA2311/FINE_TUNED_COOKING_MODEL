from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer
from huggingface_hub import login
from transformers import TrainingArguments
from datasets import load_dataset
import wandb
import os

# ==============================
# LOGIN TOKENS
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_API_TOKEN")

login(HF_TOKEN)
wandb.login(key=WANDB_TOKEN)

# ==============================
# MODEL CONFIG
# ==============================
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
max_sequence_length = 2048
load_in_4bit = True

os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"

# ==============================
# LOAD MODEL
# ==============================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_sequence_length,
    load_in_4bit=load_in_4bit,
    token=HF_TOKEN,
)

# ==============================
# LOAD DATASET
# ==============================
cook_dataset = load_dataset("EmTpro01/recipe-nlg-50k", split="train[:1000]")

EOS_TOKEN = tokenizer.eos_token

train_prompt_style = """Below is an instruction that describes a cooking task.

### Available Ingredients:
{}

### Recipe Title:
{}

### Ingredients:
{}

### Directions:
{}
"""

def preprocess_input_data(examples):
    texts = []
    for avail, title, ing, direction in zip(
        examples["available_ingredients"],
        examples["title"],
        examples["ingredients"],
        examples["directions"],
    ):
        text = train_prompt_style.format(
            avail, title, ing, direction
        ) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

finetune_dataset = cook_dataset.map(preprocess_input_data, batched=True)

# ==============================
# LORA
# ==============================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ==============================
# TRAINER
# ==============================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=finetune_dataset,
    dataset_text_field="text",
    max_seq_length=max_sequence_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=10,
        output_dir="outputs",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
    ),
)

# ==============================
# TRAIN
# ==============================
wandb.init(project="DeepSeek-Recipe-LLM")

trainer.train()

model.save_pretrained("outputs")
tokenizer.save_pretrained("outputs")

wandb.finish()

print("✅ Training Complete! Model saved in /outputs")
