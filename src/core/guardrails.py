from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailDecision:
    allowed: bool
    risk_level: str
    reason: str
    safe_reply: str | None = None
    normalized_args: dict[str, Any] = field(default_factory=dict)


# ── Prompt injection detection ────────────────────────────────────
# Detects attempts to override system instructions or extract prompts.
INJECTION_PATTERNS = (
    "ignore all previous",
    "ignore all instructions",
    "ignore your instructions",
    "ignore above instructions",
    "ignore the above",
    "disregard all previous",
    "disregard your instructions",
    "forget all previous",
    "forget your instructions",
    "you are now",
    "act as a ",
    "act as an ",
    "pretend you are",
    "new persona",
    "override your",
    "override system",
    "system prompt",
    "reveal your prompt",
    "show me your prompt",
    "print your instructions",
    "what are your instructions",
    "what is your system prompt",
    "repeat your instructions",
    "output your rules",
    "tell me your rules",
    "jailbreak",
    "dan mode",
    "developer mode",
    "do anything now",
    "roleplay as",
    "you have no restrictions",
    "no longer bound",
    "bypass your",
    "bypass safety",
    "ignore safety",
    "ignore guidelines",
    "ignore content policy",
)

# Regex patterns for more sophisticated injection attempts
INJECTION_REGEX_PATTERNS = (
    r"<\s*system\s*>",          # Fake system tags
    r"\[INST\]",                # Model instruction markers
    r"\[/INST\]",
    r"<<\s*SYS\s*>>",          # Llama system markers
    r"human:\s*assistant:",     # Role confusion
    r"```\s*system",            # Code-block system injection
)

SELF_HARM_PATTERNS = (
    "kill myself",
    "suicide",
    "end my life",
    "hurt myself",
    "die today",
)
VIOLENCE_PATTERNS = (
    "destroy my enemy",
    "harm my enemy",
    "revenge",
    "make him suffer",
    "make her suffer",
    "control someone",
    "obsession",
    "make someone love me",
    "vashikaran",
)
CURSE_PATTERNS = (
    "am i cursed",
    "black magic",
    "nazar",
    "evil eye",
    "jadu tona",
    "curse on me",
)
MEDICAL_PATTERNS = (
    "cure cancer",
    "treat cancer",
    "diagnose",
    "medical treatment",
    "which medicine",
    "disease cure",
)
LEGAL_FINANCIAL_PATTERNS = (
    "court case guarantee",
    "guarantee marriage",
    "guarantee job",
    "guarantee visa",
    "stock market",
    "trading profit",
    "lottery number",
    "legal advice",
)
SEXUAL_PATTERNS = (
    "sexual",
    "explicit",
    "nude",
    "rape",
    "abuse",
)
POLITICAL_PATTERNS = (
    "election",
    "prime minister",
    "politician",
    "president",
    "government",
    "modi",
    "trump",
    "biden",
    "rahul gandhi",
    "kejriwal",
    "yogi",
    "amit shah",
)
OUT_OF_DOMAIN_PATTERNS = (
    "debug this code",
    "write python",
    "laptop recommendation",
    "mobile app bug",
)
# Strong astrology terms — any ONE of these is enough to mark a message
# as in-domain. These words almost never appear in non-astrology contexts.
ASTROLOGY_TERMS_STRONG = {
    "astrology",
    "astrologer",
    "bracelet",
    "chart",
    "compatibility",
    "consultant",
    "dasha",
    "graha",
    "horoscope",
    "jyotish",
    "kundali",
    "kundli",
    "lagna",
    "mahadasha",
    "mala",
    "mantra",
    "matchmaking",
    "nakshatra",
    "pandit",
    "planets",
    "pooja",
    "puja",
    "rahu",
    "rashi",
    "remedies",
    "remedy",
    "rudraksha",
    "saturn",
    "spiritual",
    "spirituality",
    "zodiac",
}

# Medium-confidence terms — these indicate astrology intent when paired
# with personal intent words ("my", "I", "mujhe") or another medium term.
# These are topics people commonly consult an astrologer about.
ASTROLOGY_TERMS_MEDIUM = {
    "career",
    "destiny",
    "finance",
    "future",
    "guidance",
    "love",
    "marriage",
    "peace",
    "relationship",
}

# Weak terms — these are too generic on their own. They ONLY count
# when paired with at least one strong or medium term.
# "My work laptop" should fail, but "my work and career" should pass.
ASTROLOGY_TERMS_WEAK = {
    "birth",
    "clarity",
    "decision",
    "family",
    "health",
    "house",
    "job",
    "money",
    "moon",
    "partner",
    "planet",
    "problem",
    "stress",
    "studies",
    "study",
    "timing",
    "transit",
    "venus",
    "work",
}

PERSONAL_INTENT_WORDS = {
    "i", "me", "my", "mine", "myself",
    "mujhe", "mera", "meri", "mere", "main", "hum", "hamara",
    "aap", "aapka", "aapki",
    "should", "help", "guide", "tell", "suggest",
}

# Combined set for backward compatibility with code that checks ASTROLOGY_TERMS
ASTROLOGY_TERMS = ASTROLOGY_TERMS_STRONG | ASTROLOGY_TERMS_MEDIUM | ASTROLOGY_TERMS_WEAK
FEAR_MONETIZATION_PATTERNS = (
    "black magic",
    "curse",
    "enemy",
    "control someone",
    "make someone love me",
    "obsession",
    "guaranteed result",
    "guarantee result",
)
GUARANTEE_PATTERNS = (
    "guarantee",
    "100% result",
    "sure result",
    "fix everything",
    "must buy",
    "only this remedy will work",
)


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _contains_injection(text: str) -> bool:
    """Check for prompt injection attempts using both substring and regex patterns."""
    if _contains_any(text, INJECTION_PATTERNS):
        return True
    for pattern in INJECTION_REGEX_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def pre_scope_guardrail(message: str) -> GuardrailDecision:
    lowered = " ".join(message.lower().split())
    tokens = set(re.findall(r"[a-z0-9']+", lowered))

    if _contains_injection(lowered):
        return GuardrailDecision(
            allowed=False,
            risk_level="high",
            reason="prompt_injection",
            safe_reply=(
                "I am here for astrology and spiritual guidance. "
                "If you have a question about your chart, remedies, or life concerns, I am happy to help. 🙏"
            ),
        )

    if _contains_any(lowered, SELF_HARM_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="high",
            reason="self_harm",
            safe_reply=(
                "I am sorry this feels so overwhelming right now. Please reach out to a trusted person or local emergency or crisis support immediately. "
                "If you want spiritual comfort later, we can keep that gentle and calm."
            ),
        )
    if _contains_any(lowered, VIOLENCE_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="high",
            reason="harm_or_manipulation",
            safe_reply=(
                "I understand this feels intense, but I cannot help with harming, controlling, or forcing another person. "
                "If you want, we can shift this toward protection, emotional balance, and peaceful spiritual guidance."
            ),
        )
    if _contains_any(lowered, MEDICAL_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="high",
            reason="medical_claim",
            safe_reply=(
                "I would not treat a medical condition through astrology alone. "
                "For health concerns, please speak with a qualified doctor as well. If you want, I can still suggest a calm prayer or grounding practice for peace of mind."
            ),
        )
    if _contains_any(lowered, LEGAL_FINANCIAL_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="high",
            reason="legal_financial_certainty",
            safe_reply=(
                "I should not give guaranteed legal or financial certainty here. "
                "If you want, I can still offer calm astrology guidance about timing, mindset, and practical caution."
            ),
        )
    if _contains_any(lowered, CURSE_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="high",
            reason="curse_fear",
            safe_reply=(
                "I understand why this feels worrying, but I should not confirm curses or create fear around them. "
                "If you want, we can look at this calmly as a peace, protection, and spiritual-balance concern."
            ),
        )
    if _contains_any(lowered, SEXUAL_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="high",
            reason="sexual_or_abusive_content",
            safe_reply=(
                "I cannot help with that, but I can still support you with calm astrology or spiritual guidance if you want to shift the topic."
            ),
        )
    if _contains_any(lowered, POLITICAL_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="medium",
            reason="political_or_public_figure",
            safe_reply=(
                "I am here for personal astrology and spiritual guidance, not politics or public-figure predictions. "
                "If you want, ask about your own chart, timing, or life concerns."
            ),
        )
    if _contains_any(lowered, OUT_OF_DOMAIN_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="medium",
            reason="out_of_domain",
            safe_reply=(
                "I am here for astrology and spiritual guidance. "
                "If you want, ask about your chart, remedies, timing, relationships, career, or peace of mind."
            ),
        )

    # All hard safety checks passed — let the LLM persona handle scope.
    # The system prompt already defines an astrology-only persona, so
    # off-topic messages get redirected naturally without brittle keywords.
    return GuardrailDecision(allowed=True, risk_level="low", reason="allowed")


def tool_specific_guardrail(
    intent: str,
    message: str,
    tool_args: dict[str, Any] | None = None,
) -> GuardrailDecision:
    lowered = " ".join(message.lower().split())
    normalized_args = dict(tool_args or {})

    if _contains_any(lowered, VIOLENCE_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="high",
            reason="harm_or_manipulation",
            safe_reply=(
                "I cannot help with harmful or controlling intentions. "
                "If you want, I can suggest a simple prayer or calming spiritual practice instead."
            ),
        )
    if intent in {"recommend_product", "book_pooja"} and _contains_any(lowered, FEAR_MONETIZATION_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="high",
            reason="fear_based_monetization_block",
            safe_reply=(
                "I would not turn fear into a product or pooja recommendation. "
                "If you want, we can keep this simple and look at peaceful spiritual guidance instead."
            ),
        )
    if intent == "recommend_product" and _contains_any(lowered, MEDICAL_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="high",
            reason="medical_product_block",
            safe_reply=(
                "I would not suggest a spiritual product as a medical cure. "
                "Please speak with a doctor for treatment, and if you want I can still suggest a gentle mantra for emotional support."
            ),
        )
    if _contains_any(lowered, GUARANTEE_PATTERNS):
        return GuardrailDecision(
            allowed=False,
            risk_level="high",
            reason="guaranteed_claim_block",
            safe_reply=(
                "I should not promise guaranteed outcomes from astrology, products, or poojas. "
                "If you want, I can still guide you in a balanced and practical way."
            ),
        )
    if "search_query" in normalized_args and isinstance(normalized_args["search_query"], str):
        normalized_args["search_query"] = " ".join(normalized_args["search_query"].split()).strip()
    return GuardrailDecision(
        allowed=True,
        risk_level="low",
        reason="allowed",
        normalized_args=normalized_args,
    )


def sanitize_user_input(message: str) -> str:
    """Strip characters and patterns that could confuse the LLM into role-switching.

    This runs BEFORE the message enters any prompt template so that
    injected role markers or fake XML tags are neutralized.
    """
    sanitized = message
    # Remove fake role markers
    sanitized = re.sub(r"(human|assistant|system)\s*:", "", sanitized, flags=re.IGNORECASE)
    # Remove fake XML/instruction tags
    sanitized = re.sub(r"<\s*/?\s*(system|instruction|prompt|rule|override)\s*>", "", sanitized, flags=re.IGNORECASE)
    # Remove model-specific markers
    sanitized = re.sub(r"\[/?INST\]", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"<<\s*/?\s*SYS\s*>>", "", sanitized, flags=re.IGNORECASE)
    # Collapse excessive whitespace from removals
    sanitized = " ".join(sanitized.split()).strip()
    return sanitized or message


def final_response_guardrail(response: str) -> str:
    softened = " ".join((response or "").split())
    replacements = {
        "you are cursed": "this may feel emotionally heavy",
        "this pooja will fix everything": "this pooja is traditionally done for peace and support",
        "guaranteed marriage": "supportive for marriage-related peace and clarity",
        "your health will become bad": "this may feel draining, but for health concerns please consult a doctor as well",
        "you must buy this": "you may consider this if it resonates",
        "only this remedy will work": "this can be one simple supportive remedy",
    }
    lowered = softened.lower()
    for source, target in replacements.items():
        if source in lowered:
            softened = re.sub(re.escape(source), target, softened, flags=re.IGNORECASE)
            lowered = softened.lower()
    words = softened.split()
    if len(words) > 180:
        softened = " ".join(words[:180]).rstrip(",.;:") + "."
    return softened
