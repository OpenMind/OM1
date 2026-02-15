import logging
from typing import List, Optional


class MockTextProvider:
    """
    Singleton class to provide mock text data to ASR and other text-based inputs.

    This class serves as a central repository for test text data that can be
    used by different text-based input implementations during testing.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MockTextProvider, cls).__new__(cls)
            cls._instance.texts = []
            cls._instance.current_index = 0
            logging.info("Initialized MockTextProvider singleton")
        return cls._instance

    def load_texts(self, texts: List[str]):
        """
        Load a sequence of test texts.

        Parameters
        ----------
        texts : List[str]
            List of text strings to use for testing
        """
        self.texts = texts
        self.current_index = 0
        logging.info(f"MockTextProvider loaded {len(self.texts)} test texts")

    def get_next_text(self) -> Optional[str]:
        """
        Get the next text in the sequence.

        Returns
        -------
        Optional[str]
            Next test text or None if no more texts
        """
        if not self.texts or self.current_index >= len(self.texts):
            return None

        text = self.texts[self.current_index]
        self.current_index += 1
        return text

    def reset(self):
        """Reset the text provider to start from the first text again."""
        self.current_index = 0

    def clear(self):
        """Clear all loaded texts and reset the index."""
        self.texts = []
        self.current_index = 0


def get_text_provider() -> MockTextProvider:
    """Get the singleton text provider instance."""
    return MockTextProvider()


def get_next_text() -> Optional[str]:
    """Get the next test text."""
    provider = get_text_provider()
    return provider.get_next_text()


def clear_text_provider():
    """Clear all loaded texts."""
    provider = get_text_provider()
    provider.clear()
