from om1.privacy.anonymizer import Anonymizer

def test_email_mask():
    a = Anonymizer()
    text = "Contact: alice@example.com"
    assert a.anonymize(text) == "Contact: [EMAIL]"

def test_phone_mask():
    a = Anonymizer()
    text = "Call me at 123-456-7890"
    assert a.anonymize(text) == "Call me at [PHONE]"

def test_card_mask():
    a = Anonymizer()
    text = "Card: 1111-2222-3333-4444"
    assert a.anonymize(text) == "Card: [CARD]"
