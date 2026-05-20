# Observability

## Eval Isolation

Planner eval runs can be isolated from production by setting:

- `EVAL_MODE=1`
- `GROQ_API_KEY_EVAL`

When eval mode is enabled:

- the planner uses `GROQ_API_KEY_EVAL` when present
- Langfuse LLM traces are tagged with `environment=eval`
- an `EVAL_RUN_ID` is attached so a single eval run can be filtered cleanly

For Docker:

```bash
docker compose --profile eval up eval
```

For PowerShell:

```powershell
./scripts/eval_planner_in_docker.ps1
```
