# Digveda AI - Feature Sheet

## Implemented & Working

### Core AI Pipeline
| Feature | Status | Details |
|---------|--------|---------|
| Streaming Chat (SSE) | Live | Multi-stage pipeline: planner > guardrails > RAG > emotion > response composition. Real-time token streaming via Server-Sent Events |
| Dual Planner Architecture | Live | ConversationPlanner (JSON) + ToolCallPlanner (function calling). Conversation history-aware, pandit-philosophy prompts |
| Multi-Provider LLM | Live | Groq, Gemini, OpenAI-compatible. Separate providers for planner vs response (e.g., Gemini Flash for both) |
| Hybrid RAG | Live | Vector (pgvector) + keyword search, astro entity extraction, query expansion, reciprocal rank fusion, heuristic reranking, chart-context-aware scoring. 70 astrology documents indexed |
| Emotion Detection | Live | 8 emotion states (fearful, anxious, career_stress, health_worry, relationship_stress, confused, devotional, neutral) with intensity levels. Drives response styling |
| Guardrails (3-layer) | Live | Pre-scope (injection, self-harm, violence, medical, legal, political, sexual, out-of-domain), tool-specific (fear monetization, guarantees), output softening |
| Persona System | Live | Digveda Pandit persona: warm, traditional Vedic astrologer. Hinglish support. Emotion-adaptive tone. Chart-aware responses |
| Memory System | Live | Session-scoped recent turns + user-scoped long-term facts (cross-session). Auto fact extraction via LLM. Conversation summarization |
| Idempotent Messages | Live | Client message ID based replay. Prevents duplicate processing on retry |
| Rate Limiting | Live | Per-session/user configurable limits |

### Tools & Actions
| Feature | Status | Details |
|---------|--------|---------|
| Product Recommendation | Live | Catalog search via core-service API. Only rudraksha, mala, bracelets, pendants. Planet-to-mukhi mapping. Soft recommendations (gentle aside) + explicit recommendations. Product cards via SSE |
| Show Kundali | Live | Birth chart computation (local/remote engine). Chart summary with placements, dasha, house analysis. Kundali card via SSE |
| Matchmaking | Live | Guna milan / kundali matching. Requires both partner birth details. Summary generation. Auth required |
| Book Pooja | Live | Home puja + temple service listing from core-service catalog. Service cards via SSE |
| Confirm Booking | Live | Booking confirmation flow. Price preview, tier selection, core-service integration |
| Suggest Consultant | Live | Pandit/astrologer search and listing from core-service. Consultant cards via SSE |
| Schedule Consultation | Live | 1:1 video call booking with consultant. Date/time preference |
| Check Booking | Live | User booking history with status filter (pending, confirmed, completed, cancelled) |

### Behavioral Intelligence
| Feature | Status | Details |
|---------|--------|---------|
| Energy Flow Tracking | Live | Real-time behavior events: typing pauses, message patterns, app re-opens, session timing |
| Behavioral Scoring | Live | 6 metrics: stress, focus, emotional drift, cognitive overload, clarity, behavioral consistency |
| State Detection | Live | Emotional states (elevated_stress, uncertain, reflective), focus states (wavering, scattered), behavioral states (overthinking_loop, drained_rhythm, grounded) |
| Behavioral Context Injection | Live | Energy flow snapshot woven into system prompt for behavior-aware responses |
| Pattern Engine | Live | Recurring topic analysis across sessions. Dominant theme extraction. Topic-specific interrupt actions |

### Daily Insights & Notifications
| Feature | Status | Details |
|---------|--------|---------|
| Daily Personalized Insights | Live | Morning insight combining: birth chart + current transits + predictive insights + pattern analysis + energy flow overrides |
| Push Notification System | Live | Device token management (expo/fcm/apns). Scheduled delivery. Engagement tracking (opened/tapped/dismissed) |
| Batch Scheduler | Live | Internal endpoint triggers daily generation for all users with birth details. 50 users/batch |

### Astrology Engine
| Feature | Status | Details |
|---------|--------|---------|
| Birth Chart Computation | Live | Full Vedic chart: 12 houses, 9 planets, nakshatras, ayanamsha support, house systems |
| Dasha Periods | Live | Mahadasha + antardasha extraction and current period identification |
| Transit Analysis | Live | Current planetary positions vs natal chart. Transit impact assessment |
| Predictive Insights | Live | Future timing predictions based on chart + transits + dasha |
| Yoga Detection | Live | Astrological yoga pattern identification in birth charts |

### Infrastructure
| Feature | Status | Details |
|---------|--------|---------|
| Docker Compose Stack | Live | PostgreSQL 16 + pgvector, Redis 7, API service (uvicorn) |
| Core-Service Integration | Live | REST client for products, bookings, consultants, kundli, matchmaking |
| JWT Authentication | Live | Token validation, optional auth (guest + authenticated flows) |
| LLM Observability | Live | Langfuse integration for tracing all LLM calls |
| Prompt Versioning | Live | Content hash tracking per prompt file. Version metadata in every response |

---

## Planned / Next Up

### Short-Term (Next Sprint)
| Feature | Priority | Details |
|---------|----------|---------|
| Embedding Upgrade | High | Switch from local_hash to Gemini embeddings (gemini-embedding-001, 3072 dims). Config ready, needs reindex |
| Viral Loop / Share | High | Shareable insight cards (daily insight, kundali summary) with deep links for user acquisition |
| Kundali PDF Export | Medium | Downloadable PDF report of full birth chart analysis |
| Conversation History UI Sync | Medium | API for fetching past conversations with pagination |
| Multi-language Persona | Medium | Full Hindi persona variant (not just Hinglish mixing) |

### Medium-Term (Next Month)
| Feature | Priority | Details |
|---------|----------|---------|
| AI Reranker | High | Replace heuristic reranker with LLM-based reranking for better RAG precision |
| Proactive Notifications | High | Transit-triggered alerts ("Saturn entering your 10th house next week — here's what to prepare") |
| Consultation Follow-Up | Medium | Post-consultation summary + action items auto-generated from session |
| Product Recommendation V2 | Medium | Chart-aware product scoring: afflicted planets map directly to mukhi suggestions with confidence |
| Voice Input | Medium | Speech-to-text for message input (client-side + API support) |
| Matchmaking Report | Medium | Detailed compatibility report beyond guna score — dosha analysis, remedy suggestions |

### Long-Term (Roadmap)
| Feature | Priority | Details |
|---------|----------|---------|
| Multi-Agent Architecture | High | Specialist agents (career pandit, relationship pandit, remedy pandit) with routing |
| Kundali Comparison | Medium | Side-by-side chart comparison for family/partner analysis |
| Muhurta Engine | Medium | Auspicious timing calculator for events (marriage, griha pravesh, business start) |
| Astrological Calendar | Medium | Personalized monthly calendar with favorable/unfavorable days |
| Community Features | Low | User-to-user Q&A, pandit-verified answers, trending topics |
| Subscription Tiers | Low | Free (limited messages) → Premium (unlimited + daily insights + priority consultants) |

---

## Architecture Overview

```
Mobile App (React Native)
    |
    v
[Cloudflare Tunnel / Load Balancer]
    |
    +---> astro-chatbot-service (port 8010)
    |       |-- Planner (Gemini Flash / Groq)
    |       |-- RAG (pgvector + keyword)
    |       |-- Memory (PostgreSQL)
    |       |-- Energy Flow (behavioral scoring)
    |       |-- Guardrails (3-layer safety)
    |       |-- Response LLM (Gemini Flash / Groq)
    |       |-- Tool Executor
    |       |-- Daily Insight Scheduler
    |       +-- Push Notification Service
    |
    +---> core-service (port 8000)
            |-- Product Catalog (rudraksha, mala, bracelets)
            |-- Booking System (home puja, temple, consultants)
            |-- User Management
            |-- Kundli/Matchmaking Engine
            +-- Payment Integration
```

## Key Numbers

| Metric | Value |
|--------|-------|
| RAG Documents | 70 astrology texts |
| Planner Actions | 9 (respond_only, show_kundali, matchmaking, book_pooja, recommend_product, suggest_consultant, confirm_booking, schedule_consultation, check_booking) |
| Guardrail Patterns | 60+ (injection, safety, scope) |
| Emotion States | 8 |
| Behavioral Metrics | 6 scores + 3 state dimensions |
| Product Categories | 4 (rudraksha, mala, bracelets, pendants) |
| LLM Providers Supported | 3 (Groq, Gemini, OpenAI-compatible) |
| Database Tables | 12 |
| API Endpoints | 18 |
| Tool Definitions | 9 |
