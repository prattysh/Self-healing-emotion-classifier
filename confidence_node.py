class ConfidenceCheckNode:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def check(self, prediction_result):
        confidence = prediction_result["confidence"]
        is_confident = confidence >= self.threshold
        return {
            "is_confident": is_confident,
            "confidence": confidence,
            "label": prediction_result["label"]
        }

# Example
if __name__ == "__main__":
    dummy = {"label": "joy", "confidence": 0.54}
    node = ConfidenceCheckNode(threshold=0.7)
    result = node.check(dummy)
    print(result)
