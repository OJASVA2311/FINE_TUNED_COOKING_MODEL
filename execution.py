from unsloth import FastLanguageModel
import torch

# ==============================
# LOAD FINE-TUNED MODEL
# ==============================
model_lora, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs",  # load trained model
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model_lora)

# ==============================
# LOAD BASE MODEL
# ==============================
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(base_model)
# ==============================
# ASK FUNCTIONS
# ==============================
def ask_base(question):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = base_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = base_model.generate(**inputs, max_new_tokens=300)
    return base_tokenizer.decode(outputs[0], skip_special_tokens=True)


def ask_finetuned(question):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model_lora.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ==============================
# SESSION MEMORY
# ==============================
class RecipeSession:
    def __init__(self, max_turns=5):
        self.main_recipe = ""
        self.memory = []
        self.max_turns = max_turns

    def set_main_recipe(self, recipe):
        self.main_recipe = recipe
        self.memory = []

    def add_user(self, msg):
        self.memory.append(f"User: {msg}")

    def add_model(self, msg):
        self.memory.append(f"Chef: {msg}")

    def get_context(self):
        history = "\n".join(self.memory[-2*self.max_turns:])
        return f"Main Recipe:\n{self.main_recipe}\n\nConversation:\n{history}"

    def reset(self):
        self.main_recipe = ""
        self.memory = []


# ==============================
# RUN TEST
# ==============================
if __name__ == "__main__":

    print("BASE MODEL:\n")
    print(ask_base("How do I make butter chicken?"))

    print("\nFINE-TUNED MODEL:\n")
    recipe = ask_finetuned("How do I make butter chicken?")
    print(recipe)

    session = RecipeSession()
    session.set_main_recipe(recipe)

    followup = ask_finetuned("Can I replace cream with milk?")
    print("\nFollow-up:\n", followup)
