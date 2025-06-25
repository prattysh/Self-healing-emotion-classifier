from langgraph_dag import LangGraphDAG
from logger import log_prediction

def main():
    print("=" * 60)
    print("🧠 Self-Healing Text Classifier CLI (ATG Assignment)")
    print("Type 'exit' to quit.")
    print("=" * 60)

    dag = LangGraphDAG(threshold=0.7)

    while True:
        user_input = input("\nEnter input text:\n> ").strip()
        if user_input.lower() == "exit":
            print("👋 Exiting. Goodbye!")
            break

        result = dag.run(user_input)

        # Logging
        log_prediction(
            input_text=user_input,
            label=result["final_label"] if not result["fallback_used"] else "uncertain",
            confidence=result["confidence"],
            fallback_used=result["fallback_used"],
            final_label=result["final_label"]
        )

        print(f"\n✅ Final Label: {result['final_label']}")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print("🔁 Fallback Triggered:" if result["fallback_used"] else "✅ Prediction Accepted")

if __name__ == "__main__":
    main()
