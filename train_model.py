from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from data_preparation import load_emotion_dataset, tokenize_dataset
import torch

def get_lora_model(base_model_name, num_labels):
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],  # for DistilBERT
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    
    model = get_peft_model(base_model, lora_config)
    return model

def train_model():
    raw_dataset = load_emotion_dataset()
    tokenized_dataset = tokenize_dataset(raw_dataset)

    model = get_lora_model("distilbert-base-uncased", num_labels=6)

    training_args = TrainingArguments(
    output_dir="./model/fine_tuned",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch"
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    trainer.train()
    trainer.save_model("./model/fine_tuned")

if __name__ == "__main__":
    train_model()
