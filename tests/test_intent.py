from src.core.intent import IntentClassifier


def test_intent_classifier_detects_kundali() -> None:
    result = IntentClassifier().classify("Please read my kundali")
    assert result.name == "show_kundali"
