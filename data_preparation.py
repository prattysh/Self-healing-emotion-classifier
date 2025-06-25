from datasets import load_dataset
from transformers import AutoTokenizer

def load_emotion_dataset():
    dataset = load_dataset("emotion")  # multi-class dataset
    return dataset

def tokenize_dataset(dataset, tokenizer_name="distilbert-base-uncased", max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_fn(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=max_length)

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized

if __name__ == "__main__":
    dataset = load_emotion_dataset()
    tokenized = tokenize_dataset(dataset)
    print("Sample tokenized input:", tokenized['train'][0])
