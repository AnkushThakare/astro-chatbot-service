from __future__ import annotations

from astro_chatbot_service.models.schemas import RetrievalMatch


def build_messages(
    system_prompt: str,
    memory_turns: list[dict[str, str]],
    retrieved_docs: list[RetrievalMatch],
    user_message: str,
    astrology_summary: str | None,
) -> list[dict[str, str]]:
    sections = [system_prompt]

    if astrology_summary:
        sections.append(f"Astrology context:\n{astrology_summary}")

    if retrieved_docs:
        doc_lines = [f"- {doc.title}: {doc.excerpt}" for doc in retrieved_docs]
        sections.append("Retrieved knowledge:\n" + "\n".join(doc_lines))

    messages = [{"role": "system", "content": "\n\n".join(sections)}]
    messages.extend(memory_turns)
    messages.append({"role": "user", "content": user_message})
    return messages

