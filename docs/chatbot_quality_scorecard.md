# Chatbot Quality Scorecard

This file is the operating dashboard for Digveda chatbot quality.

Use it for three things:
- track where the chatbot stands right now
- decide whether a code or prompt change improved or damaged quality
- keep the team aligned on what matters most

## North Star

| Area | Current | Target | How We Measure It | Ship Gate |
| --- | --- | --- | --- | --- |
| Tech stack robustness | 7.5/10 | 9/10 | migrations, runtime stability, cache behavior, error rate | no critical infra regressions |
| Astrology retrieval | 8/10 | 9/10 | top-3 retrieval hit rate on benchmark prompts | >= 90% top-3 hit rate on core eval set |
| Personalization | 5/10 | 9/10 | DOB continuity, language continuity, memory continuity, chart-aware carry-over | no failures on memory-critical prompts |
| Response quality | 5/10 | 9/10 | tone score, specificity score, readability score, user preference alignment | average >= 4/5 on offline rubric |
| Birth chart reading | 6/10 | 9/10 | chart grounding, placement interpretation quality, dasha/transit integration | no hallucinated chart details |
| Hinglish quality | 8/10 | 9/10 | naturalness, script consistency, tone warmth | no awkward translation-style outputs |
| Daily engagement | 4/10 | 9/10 | daily insight open rate, return rate, repeated usage | daily insight quality benchmark passes |
| Product maturity | 4.5/10 | 9/10 | release discipline, eval coverage, observability, rollback confidence | benchmark + smoke test required before ship |

## Current Risks

| Risk | Why It Matters | Status |
| --- | --- | --- |
| Changes are made without measurement | product may get worse while feeling better in isolated examples | active |
| Response quality can drift after prompt edits | stronger tone in one path may damage safety or clarity elsewhere | active |
| Personalization is still incomplete | users notice instantly when DOB, language, or past concerns are forgotten | active |
| Retrieval can still miss intent clusters | obvious misses destroy trust even if the rest of the bot is good | active |
| Daily engagement loop is weaker than competitors | without habit-forming behavior, product quality alone will not win | active |

## Release Scorecard

Fill one row for every meaningful chatbot change.

| Date | Branch / Commit | Change Area | Benchmark Version | Passed / Total | Critical Regressions | Overall Verdict | Ship? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-20 | pending | memory + response quality + language sync | v0.1 | not run here | not verified in this shell | code reviewed only | no |
| YYYY-MM-DD |  |  |  |  |  |  |  |
| YYYY-MM-DD |  |  |  |  |  |  |  |
| YYYY-MM-DD |  |  |  |  |  |  |  |

## Change Review Rubric

Score each benchmark output on a `1 to 5` scale.

| Dimension | 1 | 3 | 5 |
| --- | --- | --- | --- |
| Retrieval relevance | wrong or weak context | partly relevant | clearly grounded in the best context |
| Astrology specificity | generic advice | some chart/remedy detail | precise planets, houses, dasha, timing, remedy logic |
| Personalization | sounds like first-time chat | some continuity | clearly remembers language, concern, and profile context |
| Tone attractiveness | flat or robotic | usable | warm, sharp, memorable, human |
| Business correctness | invents products/services or misses app context | mostly correct | fully aligned with Digveda catalog, pooja, and consultation flows |
| Safety and restraint | fear-heavy or overclaiming | mostly safe | calm, bounded, trustworthy |

## Critical Regression Rules

Do not ship if any of these fail:
- user already gave DOB, but chatbot asks again without reason
- wrong language flow for obvious Hinglish or plain English user
- invented product, consultant, pooja, or app capability
- chart reading mentions placements not present in chart context
- obvious retrieval miss on a core topic like `sade sati`, `mangal dosha`, `consultation flow`, or `Digveda products`
- response becomes shorter but meaningfully weaker after a prompt change

## Required Evidence Before Shipping

Every meaningful chatbot release should have:
- benchmark run result
- 5 manual spot-check chats
- 1 memory continuity check
- 1 Digveda business-flow check
- 1 chart-reading check
- rollback note if the release touches prompts, retrieval, or orchestration

## Suggested Benchmarks to Track

Track these headline numbers across versions:
- `core_eval_pass_rate`
- `top3_retrieval_hit_rate`
- `memory_continuity_pass_rate`
- `language_alignment_pass_rate`
- `business_policy_pass_rate`
- `tone_score_avg`
- `specificity_score_avg`

## Progress Bands

| Score Band | Meaning |
| --- | --- |
| 4 to 5 | early-stage, inconsistent |
| 6 to 7 | strong prototype |
| 7.5 to 8.5 | product-grade and improving with discipline |
| 9 to 10 | category-level quality with stable eval-driven iteration |

## Owner Checklist For Any Change

- What exactly changed?
- Which benchmark prompts should improve because of this?
- Which benchmark prompts might regress because of this?
- Was benchmark re-run after the change?
- Did any critical regression appear?
- If shipped, how do we know it helped real users?
