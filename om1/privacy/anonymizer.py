import re

class Anonymizer:
    def __init__(self, level="standard"):
        self.level = level

    def anonymize(self, text: str) -> str:
        # Email mask
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)
        # Phone mask
        text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
        # Credit card mask
        text = re.sub(r"\b\d{4}-\d{4}-\d{4}-\d{4}\b", "[CARD]", text)
        return text

# Example usage
if __name__ == "__main__":
    anon = Anonymizer()
    sample_text = "User john.doe@gmail.com paid with card 4242-4242-4242-4242"
    print(anon.anonymize(sample_text))
