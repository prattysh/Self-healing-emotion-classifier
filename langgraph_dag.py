from inference_node import InferenceNode
from confidence_node import ConfidenceCheckNode
from fallback_node import FallbackNode

class LangGraphDAG:
    def __init__(self, threshold=0.7):
        self.inference_node = InferenceNode()
        self.confidence_node = ConfidenceCheckNode(threshold=threshold)
        self.fallback_node = FallbackNode()

    def run(self, input_text):
        print(f"\nInput: {input_text}\n")

        # Step 1: Inference
        result = self.inference_node.predict(input_text)
        print(f"[InferenceNode] Predicted label: {result['label']} | Confidence: {result['confidence']:.2f}")

        # Step 2: Confidence Check
        confidence_result = self.confidence_node.check(result)
        if confidence_result["is_confident"]:
            print("[ConfidenceCheckNode] Confidence is sufficient. Accepting prediction.")
            return {
                "final_label": result["label"],
                "confidence": result["confidence"],
                "fallback_used": False
            }
        else:
            print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")

            # Step 3: Fallback
            corrected_label = self.fallback_node.clarify(input_text, result["label"])
            return {
                "final_label": corrected_label,
                "confidence": result["confidence"],
                "fallback_used": True
            }

# Example Test
if __name__ == "__main__":
    dag = LangGraphDAG()
    user_input = input("\nEnter your text:\n> ")
    output = dag.run(user_input)
    print("\nFinal Output:", output)
