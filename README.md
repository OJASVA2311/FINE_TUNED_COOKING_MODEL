#  Fine-Tuned Cooking Model (DeepSeek + LoRA)

A parameter-efficient fine-tuned Large Language Model built on **DeepSeek-R1-Distill-Llama-8B** for generating high-quality cooking recipes and answering culinary follow-up questions.

This project demonstrates:
- Large Language Model fine-tuning
- LoRA (Low-Rank Adaptation)
- 4-bit quantization
- Dataset preprocessing
- Structured training pipeline
- Memory-based follow-up conversation

---

 Project Overview

This project fine-tunes **DeepSeek-R1-Distill-Llama-8B** on a cooking dataset to:

- Generate structured recipes
- Provide detailed cooking instructions
- Answer ingredient substitution questions
- Maintain short conversational memory for follow-ups

Fine-tuning is performed using:
- Unsloth (efficient LLM training)
- LoRA (parameter-efficient fine-tuning)
- 4-bit quantization (memory optimization)
- Weights & Biases for experiment tracking

---

## 🧠 Model Details

- **Base Model:** DeepSeek-R1-Distill-Llama-8B
- **Fine-Tuning Method:** LoRA
- **Quantization:** 4-bit
- **Sequence Length:** 2048
- **Optimizer:** AdamW 8-bit
- **Framework:** PyTorch + Transformers + TRL

---

## 📂 Project Structure

