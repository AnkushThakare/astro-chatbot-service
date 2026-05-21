# Chatbot Improvement Roadmap

This roadmap exists to keep Digveda moving forward without getting lost inside a 30k-line codebase.

The rule is simple:
- do not improve everything at once
- improve by workstream
- measure each workstream before moving on

## Guiding Principle

The codebase is already large enough that random prompt edits and broad refactors are risky.

So the forward path should be:
1. stabilize the highest-risk foundations
2. build evaluation discipline
3. deepen chart intelligence and personalization
4. improve engagement and retention
5. polish operations and release confidence

## Phase 1: Stabilize

Goal: remove the obvious trust-breaking failures.

Focus areas:
- DOB persistence
- language persistence
- memory continuity
- Digveda business-flow retrieval
- no fake product or service mentions

Definition of done:
- all memory-critical benchmark prompts pass
- no obvious consultation / product / pooja routing failures
- no regression in existing retrieval quality

## Phase 2: Measure

Goal: stop guessing whether changes are good.

Focus areas:
- benchmark dataset
- release scorecard
- offline scoring rubric
- regression checklist
- repeatable test workflow in Docker or CI

Definition of done:
- every chatbot change is tied to a benchmark run
- no major prompt or orchestration change ships without evaluation

## Phase 3: Deepen Intelligence

Goal: make the assistant feel genuinely sharp, not just well-retrieved.

Focus areas:
- chart reading depth
- dasha + transit + placement synthesis
- better use of user memory and behavior signals
- stronger “pattern” recognition without sounding spooky
- better answer-depth selection based on user style

Definition of done:
- chart-based answers consistently feel more insightful than generic astrology apps
- personalization score crosses `7.5/10`

## Phase 4: Build Habit Loops

Goal: become a product users return to, not just a one-time Q&A bot.

Focus areas:
- daily insights quality
- follow-up question design
- session continuity
- emotionally sticky but respectful tone
- better daily/weekly engagement surfaces inside Digveda

Definition of done:
- daily engagement prompts feel useful, timely, and personal
- repeated usage improves in product analytics

## Phase 5: Operational Maturity

Goal: ship faster with lower risk.

Focus areas:
- observability
- benchmark automation
- deploy checklist
- rollback checklist
- prompt version tracking

Definition of done:
- team can tell within one release whether a change improved quality
- rollback path is clear for prompt, retrieval, or orchestration regressions

## Workstreams

Use these as parallel tracks instead of one giant refactor.

### Workstream A: Personalization

Targets:
- user profile continuity
- preferred language
- answer depth preference
- recurring concern memory
- chart availability awareness

Do next:
- extend profile persistence beyond DOB and language
- add a compact user profile summary to system context
- benchmark follow-up behavior across sessions

### Workstream B: Response Quality

Targets:
- stronger openers
- more specific astrology language
- better emotional pacing
- cleaner Hinglish
- less robotic repetition

Do next:
- build a tone benchmark set
- compare short vs medium answer lengths
- audit repetitive phrases across prompts

### Workstream C: Retrieval and Knowledge

Targets:
- fewer obvious misses
- stronger Digveda business knowledge
- stronger house/planet/yoga/dosha coverage
- chart-aware retrieval quality

Do next:
- expand eval set by topic
- track top-3 retrieval hit rate
- fix weak query families one cluster at a time

### Workstream D: Product Maturity

Targets:
- clear release discipline
- stable migrations
- observability on failures
- reproducible tests

Do next:
- ensure test runner works in Docker for chatbot evals
- add benchmark run instructions to README or docs
- define release gates

## What Not To Do

Avoid these traps:
- rewriting multiple prompts and retrieval logic together without eval
- changing memory, persona, and routing in one commit without isolation
- trusting a few good chats as proof of improvement
- trying to refactor broad architecture before the benchmark system exists

## Decision Framework For The 30k-Line Codebase

Before touching any module, ask:
- Is this a foundation problem, a quality problem, or a retention problem?
- What benchmark should improve if this change works?
- What existing behavior could break?
- Is this change isolated enough to review confidently?

If the answer is unclear, the change is probably too broad.

## Suggested Next 4 Weeks

### Week 1
- finalize benchmark dataset v0.1
- start filling the quality scorecard after each change
- verify memory, DOB, language, and Digveda business flows

### Week 2
- improve chart-reading prompt behavior
- add benchmark cases for D1, D9, dasha, transits, doshas
- fix top retrieval misses

### Week 3
- improve daily insight quality
- improve follow-up behavior and session continuity
- start measuring tone consistency

### Week 4
- lock a release process
- automate benchmark execution
- review where scores moved and set new targets
