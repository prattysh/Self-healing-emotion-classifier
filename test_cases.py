from inference_node import InferenceNode

node = InferenceNode()

inputs = [
    "I just feel so empty and hopeless right now.",
    "This is absolutely unacceptable. I’m furious!",
    "I’m terrified something bad is going to happen.",
    "I can’t stop smiling today, everything feels perfect!",
    "I’ve never felt so close to someone before.",
    "Wow, I didn’t expect that at all!",
    "I'm really nervous about my presentation tomorrow.",
    "I miss the old days and feel so low right now.",
    "He yelled at me for no reason at all!",
    "I feel completely safe and happy with my friends."
]

for text in inputs:
    result = node.predict(text)
    print(f"Input: {text}")
    print(f"  → Predicted Label: {result['label']} | Confidence: {result['confidence']:.2f}\n")
