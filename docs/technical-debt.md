# Technical Debt Register

| ID | Item | Impact | Priority | Status | Next action |
| --- | --- | --- | --- | --- | --- |
| TD-001 | No browser-level E2E suite for the frontend-to-backend journey | Regression risk at the service boundary | High | Open | Add Playwright tests in the frontend repository |
| TD-002 | No representative Render capacity baseline | Free-tier cold-start and saturation behavior is unmeasured | Medium | Open | Run controlled local and isolated-deployment tests |
| TD-003 | No durable p95 latency aggregation | Production latency trends require log export and analysis | Medium | Open | Evaluate approved log aggregation or scheduled reports |
| TD-004 | External-provider contract coverage is limited | Provider schema changes may surface at runtime | Medium | Open | Add mocked contract fixtures |
| TD-005 | Some application type-check findings are outside the current CI gate | Static defects may remain undetected | Medium | Open | Resolve or explicitly baseline the remaining findings |
