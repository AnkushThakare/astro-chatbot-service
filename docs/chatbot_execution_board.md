# Chatbot Execution Board

This file exists because planning is not the problem anymore.

The real problem is:
- too many ideas at once
- too many moving parts in a 30k-line codebase
- no hard rule for what counts as done
- no forced check that a change actually improved quality

So this document is the execution system.

## Core Rule

Do not work on more than `one primary chatbot quality goal` at a time.

Bad:
- fix memory
- rewrite RAG
- improve tone
- change chart logic
- improve pooja flow

Good:
- this week we improve `memory continuity`
- we measure it
- we ship only if the benchmark improved

## Weekly Operating Model

Every week should have:
- `1 primary goal`
- `1 success metric`
- `3 tasks max`
- `1 owner`
- `1 benchmark checkpoint`

If you cannot describe the week in those five things, the scope is too wide.

## Now / Next / Later

### Now

Use this for the only work that is allowed to move this week.

| Goal | Owner | Metric | Target | Status |
| --- | --- | --- | --- | --- |
| Build benchmark discipline and stop blind changes | team | benchmark workflow exists and is used before chatbot releases | benchmark run required before ship | active |

### Next

Only start after `Now` is done.

| Goal | Why It Matters | Blocked By |
| --- | --- | --- |
| Improve chart-reading depth | major gap vs strong competitors | benchmark system must be working |
| Improve daily engagement quality | weak retention / habit loop today | benchmark system must be working |
| Expand Digveda business-flow reliability | protects conversion and trust | consultation / product eval coverage |

### Later

| Goal | Why Not Now |
| --- | --- |
| broad architecture cleanup | too risky without stronger eval discipline |
| large prompt rewrite | likely to create regressions before measurement is mature |
| multi-module refactor | too hard to verify quickly |

## Execution Rules

### Rule 1: One Change Theme Per Cycle

Each change batch must belong to one theme:
- memory
- response quality
- retrieval
- chart reading
- business flow
- engagement

Do not mix themes in one cycle unless the benchmark proves safety.

### Rule 2: Every Task Needs a Done Definition

Bad task:
- improve personalization

Good task:
- store and reuse user language preference across sessions
- benchmark cases `memory_language_hinglish_001` and `memory_dob_reuse_001` pass

### Rule 3: No Work Without a Benchmark Link

Every task must list:
- which eval cases should improve
- which cases might regress

If that is unclear, the task is not ready.

### Rule 4: WIP Limit = 3

Never have more than 3 active chatbot tasks.

Recommended split:
- 1 execution task
- 1 verification task
- 1 documentation or cleanup task

### Rule 5: Stop Measuring by “Feel”

Do not say:
- this looks better
- I think tone improved
- RAG feels stronger

Say:
- pass rate improved from X to Y
- critical regressions stayed at 0
- tone average increased from 3.1 to 3.8

## Weekly Sprint Template

Copy this section each week.

### Sprint Window

- Week of:
- Owner:
- Primary goal:
- Why this goal now:

### Success Metric

- Main metric:
- Baseline:
- Target:

### Allowed Tasks

| Task | Why | Eval Cases | Done When | Status |
| --- | --- | --- | --- | --- |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

### Not Allowed This Week

List the tempting side work you are explicitly rejecting.

- 
- 
- 

### End-of-Week Review

- What changed:
- What benchmark improved:
- What regressed:
- What should be reverted:
- What becomes next week’s priority:

## Recommended First 4 Execution Cycles

### Cycle 1: Benchmark Discipline

Goal:
- make the benchmark workflow real

Tasks:
- finalize benchmark dataset v0.1
- add runner script
- produce first score report

Done when:
- benchmark can be run on demand
- release scorecard has first real result

### Cycle 2: Memory Reliability

Goal:
- stop trust-breaking continuity failures

Tasks:
- verify DOB reuse
- verify language reuse
- verify concern continuity

Done when:
- memory-critical cases pass consistently

### Cycle 3: Response Quality

Goal:
- make replies feel sharper and more personal without damaging safety

Tasks:
- improve opener quality
- reduce robotic phrasing
- improve answer-depth matching

Done when:
- tone and personalization scores improve without policy regressions

### Cycle 4: Chart Reading Depth

Goal:
- beat generic astrology apps on actual reading quality

Tasks:
- improve placement interpretation
- improve dasha/transit blending
- improve chart-specific remedy logic

Done when:
- chart benchmark cases clearly outperform current baseline

## Anti-Failure Checklist

Use this before starting any new work.

- Is this the one primary goal for the cycle?
- Is there a measurable success metric?
- Is there a benchmark linked to the change?
- Is the task small enough to finish this cycle?
- If it works, will the scorecard move?
- If it fails, can we revert safely?

If any answer is `no`, do not start the work yet.

## Blunt Truth

You are not failing because the roadmap is missing.

You are failing because:
- scope is too wide
- change batches are too mixed
- results are not measured tightly enough

This board fixes that only if you actually follow it.
