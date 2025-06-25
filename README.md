# Self-Healing Text Classifier (ATG Assignment)

This project is a command-line based Emotion Classifier powered by a fine-tuned DistilBERT model with efficient LoRA (Low-Rank Adaptation) adapters. It detects emotions like joy, sadness, fear, anger, surprise, and love from natural language input. 

What makes this system unique is its "self-healing design" — implemented using a LangGraph-based DAG (Directed Acyclic Graph) — that can intelligently fall back or reclassify when confidence is low. This enables more robust, real-time emotion detection even in uncertain cases.

Built with:
-  Transformers (`distilbert-base-uncased`)
-  LoRA (PEFT) for lightweight fine-tuning
-  LangGraph for self-healing inference flow
-  PyTorch backend for fast inference
-  CLI for direct user interaction

The model is fine-tuned on emotion-labeled text data and can be used in real-world scenarios like chatbots, social media monitoring, or sentiment-aware recommendation systems.

---

## Project Structure

```
├── Cli.py                 # Main CLI application
├── confidence_node.py     # Accepts/rejects prediction based on confidence
├── fallback_node.py       # Provides fallback if confidence is low
├── inference_node.py      # Loads the fine-tuned model and makes predictions
├── langgraph_dag.py       # Defines the LangGraph DAG flow
├── train_model.py         # Script to fine-tune DistilBERT with LoRA
├── data_preparation.py    # (optional - for future dataset prep)
├── test_cases.py          # Batch script to test model on sample inputs
├── requirements.txt       # Required dependencies
├── model/
│   └── fine_tuned/        # Saved trained model + LoRA adapters
├── logs/                  # TensorBoard logs from training
│   ├── training_summary.tfevents.* (TensorBoard logs)
│   └── session_logs.txt   (summary/log)
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This saves the LoRA fine-tuned model to `./model/fine_tuned`.

### 3. Run the CLI

```bash
python Cli.py
```

### 4. Run Batch Test Cases

```bash
python test_cases.py
```

This runs 10 sample emotional inputs and prints their predicted labels and confidence scores.

---

## Features

-  DistilBERT + LoRA (efficient fine-tuning)
-  LangGraph DAG for self-healing flow
-  Confidence check node
-  CLI interaction with real-time inference
-  Batch test script for quick evaluation

---

##  Training Summary

```json
{
  "train_loss": 0.664,
  "eval_loss": 0.434,
  "train_runtime": "3.3 hours"
}
```

TensorBoard logs are stored in `./logs/` and can be visualized if needed.

---

## 🔍 Supported Labels

- sadness
- joy
- love
- anger
- fear
- surprise

---

## 📌 Notes

- `logs/` folder contains TensorBoard events
- `model/fine_tuned/` must exist before running CLI