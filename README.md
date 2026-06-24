# Testcase-Generation
This project demonstrates how to build a lightweight, efficient AI system that automatically generates system test plans from high-level instructions using a fine-tuned Large Language Model (LLM).

It leverages:

Mistral-7B-Instruct as the base model
LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
4-bit quantization (BitsAndBytes) to run large models on limited hardware
FastAPI to expose the model as an API

The system is designed for software testing teams, QA engineers, and automation specialists who want to accelerate test design and reduce manual documentation effort.

✨ ##Features
✅ Generate test plans from simple natural language instructions
✅ Lightweight fine-tuning using LoRA (no need to retrain full model)
✅ Memory-efficient execution with 4-bit quantization
✅ API-ready with FastAPI backend
✅ Built on Mistral-7B-Instruct, a powerful open-source LLM
✅ Custom dataset-driven learning

🏗️ ##Project Architecture

.
├── testplangeneration.py   # Model training, fine-tuning, inference
├── connection.py           # FastAPI service
├── dataset.json            # Training dataset (instruction-response format)
├── qa-assistant-lora/      # Saved LoRA weights


