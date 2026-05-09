from src.core.intent import IntentClassifier


def test_intent_classifier_detects_kundali() -> None:
    result = IntentClassifier().classify("Please read my kundali")
    assert result.name == "show_kundali"


def test_intent_classifier_detects_matchmaking() -> None:
    result = IntentClassifier().classify("Can you do a guna milan compatibility check?")
    assert result.name == "matchmaking"


def test_intent_classifier_detects_booking() -> None:
    result = IntentClassifier().classify("I want to book a home puja for next week")
    assert result.name == "book_pooja"
