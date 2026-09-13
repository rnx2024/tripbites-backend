# Technical Debt Register

| ID | Item | Impact | Priority | Status | Next action |
| --- | --- | --- | --- | --- | --- |
| TD-001 | No cross-repository browser E2E suite against a running backend deployment | Regression risk at the service boundary | High | Open | Add an explicitly configured frontend/backend integration environment |
| TD-002 | No representative Render capacity baseline | Free-tier cold-start and saturation behavior is unmeasured | Medium | Open | Run controlled local and isolated-deployment tests |
| TD-003 | No durable p95 latency aggregation | Production latency trends require log export and analysis | Medium | Open | Evaluate approved log aggregation or scheduled reports |
| TD-004 | External-provider contract coverage is limited | Provider schema changes may surface at runtime | Medium | Open | Add mocked contract fixtures |
| TD-005 | Mypy is configured but not documented as a required CI gate | Static defects may remain undetected | Medium | Open | Decide whether `uv run mypy app` belongs in the CI quality gate |
