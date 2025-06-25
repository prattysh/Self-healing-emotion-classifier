from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import torch
import torch.nn.functional as F

class InferenceNode:
    def __init__(self, model_path="./model/fine_tuned", tokenizer_name="distilbert-base-uncased", label_list=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ✅ Load LoRA adapter config
        config = PeftConfig.from_pretrained(model_path)

        # ✅ Load base model with correct label count (6 labels)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=6  # Make sure this matches your training!
        )

        # ✅ Apply LoRA weights to base model
        self.model = PeftModel.from_pretrained(base_model, model_path).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label_list = label_list or ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            confidence, pred_id = torch.max(probs, dim=1)

        label = self.label_list[pred_id.item()]
        return {
            "label": label,
            "confidence": confidence.item(),
            "probs": probs.cpu().numpy()[0]
        }

# Test
if __name__ == "__main__":
    node = InferenceNode()
    result = node.predict("I'm feeling kind of sad and disappointed today.")
    print(f"Predicted Label: {result['label']} | Confidence: {result['confidence']:.2f}")
