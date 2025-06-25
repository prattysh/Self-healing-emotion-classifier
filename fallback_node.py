class FallbackNode:
    def __init__(self):
        pass

    def clarify(self, input_text, current_label):
        print(f"\n[FallbackNode] Low confidence in prediction: '{current_label}'.")
        print(f"Original input: \"{input_text}\"")
        clarification = input("Could you clarify your intent? What emotion were you expressing?\n> ")
        return clarification

# Example
if __name__ == "__main__":
    fallback = FallbackNode()
    final_label = fallback.clarify("I'm not sure if I was happy or nervous.", "joy")
    print("Final label (user clarified):", final_label)
