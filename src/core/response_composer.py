from __future__ import annotations

from typing import Any

from src.core.emotion import EmotionResult


def build_style_instruction(
    *,
    message: str,
    emotion: EmotionResult,
    route: str,
    intent: str,
    tool_outputs: list[dict[str, Any]],
) -> str:
    intensity = getattr(emotion, "intensity", "low")
    lines = [
        "Sound like a calm, traditional Vedic astrologer speaking directly to one person.",
        "Keep the response warm, human, grounded, gently authoritative, and emotionally precise, never robotic.",
        "Open with one line that feels personally observed, not generic. Name the core tension, timing, or repeating loop quickly.",
        "Response structure: (1) astrological insight about the user's concern using SPECIFIC details from retrieved knowledge, (2) what it means practically, (3) one next step or remedy — only mention a product if naturally relevant and it was returned by the catalog.",
        "CRITICAL: Ground your answer in the retrieved knowledge. Use specific planet names, house numbers, dasha periods, and remedy details from the knowledge — never improvise generic advice when specifics are available.",
        "Use plain, natural phrases such as 'This usually shows', 'This period can bring', or 'A simple remedy is' when they fit naturally, but vary sentence rhythm so the answer does not sound templated.",
        "Stay focused. Default to 90 to 160 words, prefer 3 to 6 sentences. Go shorter when the user is emotionally overloaded. Stay under 220 words unless the user explicitly asks for depth.",
        "Avoid phrases like 'As an AI', 'Based on the provided context', or long numbered lists.",
        "Do not sound like customer support, a report, or a UI wrapper around tools.",
        "Avoid stiff phrases like 'I have prepared' or repeated 'If you want, I can' constructions.",
        "When mentioning planets, houses, dasha, or transits, explain the meaning in plain language before sounding technical.",
        "Prefer concrete imagery over bland reassurance. Words like delay, stop-start momentum, emotional heaviness, pressure, relief, or opening are better than vague positivity.",
        "Do not create fear. Do not make guaranteed claims.",
        "Never lead or end with a product suggestion. If a product is mentioned, weave it in as a quiet aside, not as the main recommendation.",
    ]
    _emotion_style: dict[str, list[str]] = {
        "fearful": [
            "User sounds fearful — keep response SHORT (under 60 words), calming, grounded.",
            "One gentle reassurance sentence first. Avoid dramatic planetary language.",
            "Focus on what IS stable. End with one calming practice (mantra or breathing).",
            "Do NOT list multiple problems or negative effects.",
        ],
        "anxious": [
            "User sounds anxious — keep language steady and measured.",
            "Brief reassurance, then ONE clear practical step. Avoid listing multiple concerns.",
        ],
        "career_stress": [
            "User is stressed about career — be practical and action-oriented.",
            "Structure: (1) acknowledge stress briefly, (2) specific timing insight, (3) one concrete step they can take this week.",
            "Avoid vague spiritual advice. Be direct about timing and actionable steps.",
        ],
        "health_worry": [
            "User worried about health — be gentle but clear.",
            "Brief reassurance. Give astrological wellness perspective.",
            "Remind that astrology supports but does not replace medical advice. Suggest one calming practice.",
        ],
        "relationship_stress": [
            "User stressed about relationships — be empathetic and warm.",
            "Give specific astrological insight about relationship dynamics. One practical and one spiritual step.",
        ],
        "confused": [
            "User sounds confused — give a STRUCTURED response.",
            "Use 2-3 short numbered points. Start by narrowing the topic. Be specific, avoid broad theory.",
        ],
        "devotional": [
            "Match their spiritual energy with warmth and reverence.",
            "Include one simple mantra or prayer when relevant.",
        ],
    }
    lines.extend(_emotion_style.get(emotion.emotion, []))
    if intensity == "high":
        lines.append("Keep response under 80 words — shorter is better when emotion is high.")
    if route == "FAST_CHAT":
        lines.append(
            "Structure: (1) one direct opening line, (2) astrological insight using SPECIFIC details from retrieved knowledge "
            "(name the planet, house, dasha, or transit — not vague references), "
            "(3) what it means in plain terms, (4) one practical spiritual step with a specific remedy from the knowledge. "
            "Products only if user asked."
        )
    elif route == "TOOL_FLOW":
        lines.append(
            "Structure: (1) warm acknowledgement, (2) astrological context for why this matters, "
            "(3) short result summary referencing items shown, (4) one soft next step. "
            "Do NOT just announce tool results — give the astrology reasoning first."
        )
    elif route == "CLARIFICATION":
        lines.append("Ask only one natural next question.")
    if tool_outputs:
        lines.append("Do not repeat every structured item in prose. Let the structured cards carry the detail.")
    if intent == "show_kundali":
        lines.append("Answer the user's actual concern from the chart perspective, not just that the kundali is ready.")
    if "?" not in message and route != "CLARIFICATION":
        lines.append("Do not force a question at the end.")
    return "\n".join(lines)


def compose_blocked_reply(reason: str, emotion: EmotionResult) -> str:
    reassurance = "I understand why this feels worrying." if emotion.emotion in {"fearful", "anxious"} else "I understand."
    if reason == "curse_fear":
        return (
            f"{reassurance} I would not call this a curse or create fear around it. "
            "Many times such feelings come during emotionally heavy phases. Keep it simple for now: chant 'Om Namah Shivaya' calmly or sit quietly in prayer for a few minutes. "
            "If you want, I can guide you from a chart perspective without fear."
        )
    if reason in {"harm_or_manipulation", "fear_based_monetization_block"}:
        return (
            f"{reassurance} I cannot help with harming, forcing, or controlling anyone. "
            "If you want, we can shift this toward protection, inner stability, and peaceful spiritual guidance."
        )
    if reason == "medical_claim":
        return (
            f"{reassurance} I would not use astrology as a medical cure or diagnosis. "
            "Please speak with a doctor for treatment, and if you want I can still suggest a calm mantra or grounding practice for peace."
        )
    if reason == "legal_financial_certainty":
        return (
            f"{reassurance} I should not give guaranteed legal or financial certainty here. "
            "If you want, I can still offer calm astrology guidance about timing, mindset, and practical caution."
        )
    return (
        f"{reassurance} I am here for personal astrology and spiritual guidance. "
        "Ask me about your chart, remedies, timing, peace of mind, relationships, or career."
    )


def compose_clarification_reply(intent: str, missing_slots: list[str], emotion: EmotionResult) -> str:
    reassurance = "I understand." if emotion.emotion == "calm" else "I understand this matters."
    if intent == "show_kundali":
        return f"{reassurance} To check this properly, please share your date of birth first."
    if intent == "matchmaking":
        return f"{reassurance} To check compatibility properly, please share your partner's birth details first."
    if intent == "recommend_product":
        return f"{reassurance} Tell me what kind of support you want first, for example peace, focus, protection, or career clarity."
    if intent == "suggest_consultant":
        return f"{reassurance} Tell me which area you want guidance in first, such as career, marriage, or peace of mind."
    if intent == "book_pooja":
        return f"{reassurance} Tell me which puja or purpose you want to book first."
    if missing_slots:
        return f"{reassurance} Please share {missing_slots[0].replace('_', ' ')} first."
    return f"{reassurance} Tell me a little more so I can guide you properly."


def build_cards(tool_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for output in tool_outputs:
        tool = output.get("tool")
        if tool == "recommend_product":
            for item in output.get("items", [])[:3]:
                price_paise = item.get("starting_price_paise")
                price_display = f"₹{price_paise // 100}" if isinstance(price_paise, int) else None
                cards.append(
                    {
                        "type": "product",
                        "id": item.get("id"),
                        "slug": item.get("slug"),
                        "title": item.get("name"),
                        "subtitle": price_display or "Spiritual product",
                        "price_paise": price_paise,
                        "image_url": item.get("image_url"),
                        "cta": "View product",
                    }
                )
        elif tool == "book_pooja":
            for item in output.get("home_puja_services", [])[:2]:
                cards.append(
                    {
                        "type": "service",
                        "id": item.get("id"),
                        "title": item.get("name"),
                        "subtitle": "Home puja service",
                        "image_url": None,
                        "cta": "View service",
                    }
                )
            for item in output.get("temple_services", [])[:2]:
                cards.append(
                    {
                        "type": "service",
                        "id": item.get("id"),
                        "title": item.get("name"),
                        "subtitle": "Temple service",
                        "image_url": None,
                        "cta": "View service",
                    }
                )
            for item in output.get("pandits", [])[:2]:
                cards.append(
                    {
                        "type": "consultant",
                        "id": item.get("id"),
                        "title": item.get("name"),
                        "subtitle": "Available pandit",
                        "image_url": item.get("photo_url"),
                        "cta": "View pandit",
                    }
                )
        elif tool == "suggest_consultant":
            for item in output.get("items", [])[:3]:
                cards.append(
                    {
                        "type": "consultant",
                        "id": item.get("id"),
                        "title": item.get("name"),
                        "subtitle": ", ".join(item.get("specialties") or []) if isinstance(item.get("specialties"), list) else "Astrology consultant",
                        "image_url": item.get("default_photo_url"),
                        "cta": "View pandit",
                    }
                )
        elif tool == "show_kundali":
            cards.append(
                {
                    "type": "kundali",
                    "id": "kundali-summary",
                    "title": "Kundali Summary",
                    "subtitle": output.get("summary"),
                    "image_url": None,
                    "cta": "View chart",
                }
            )
        elif tool == "matchmaking":
            cards.append(
                {
                    "type": "matchmaking",
                    "id": "matchmaking-summary",
                    "title": "Matchmaking Result",
                    "subtitle": output.get("summary"),
                    "image_url": None,
                    "cta": "View compatibility",
                }
            )
    return cards


def normalize_tool_outputs(tool_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for output in tool_outputs:
        tool = output.get("tool")
        if tool == "recommend_product":
            normalized.append(
                {
                    "type": "product",
                    "reason": output.get("search_query") or output.get("summary") or "product support",
                    "items": [
                        {
                            "id": item.get("id"),
                            "name": item.get("name"),
                        }
                        for item in output.get("items", [])[:3]
                    ],
                }
            )
        elif tool == "book_pooja":
            normalized.append(
                {
                    "type": "booking",
                    "reason": output.get("summary") or "pooja support",
                    "items": [
                        {
                            "id": item.get("id"),
                            "name": item.get("name"),
                        }
                        for bucket in ("home_puja_services", "temple_services")
                        for item in output.get(bucket, [])[:2]
                    ],
                }
            )
        elif tool == "suggest_consultant":
            normalized.append(
                {
                    "type": "consultation",
                    "reason": output.get("summary") or "consultation support",
                    "items": [
                        {
                            "id": item.get("id"),
                            "name": item.get("name"),
                        }
                        for item in output.get("items", [])[:3]
                    ],
                }
            )
        elif tool == "show_kundali":
            normalized.append(
                {
                    "type": "kundali",
                    "reason": output.get("summary") or "kundali summary",
                    "items": [],
                }
            )
        elif tool == "matchmaking":
            normalized.append(
                {
                    "type": "matchmaking",
                    "reason": output.get("summary") or "matchmaking summary",
                    "items": [],
                }
            )
    return normalized
