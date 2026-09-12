# Operations Runbooks

## Render service is not responding

1. Check `/health/live`.
2. Review Render logs for startup, port, and dependency errors.
3. Check the deployment version and recent configuration changes.
4. Confirm the process is using Render's `$PORT`.
5. Recheck `/health/ready` after recovery.

## Redis is unavailable

1. Check `/health/ready`.
2. Confirm the configured Redis service and connection URL in Render.
3. Review logs for the safe Redis target and error category.
4. Restore Redis availability before declaring the backend ready.

## Weather, news, or LLM provider failure

1. Check the relevant structured error logs.
2. Confirm the provider is not rate-limiting or timing out.
3. Verify the API key configuration without printing its value.
4. Use the application's bounded retry and safe fallback behavior.
5. Re-test the affected endpoint after the provider recovers.

## Incorrect news source or unsupported claim

1. Capture the response, timestamp, request ID, and deployment version without exposing secrets.
2. Confirm the requested location and the source item used.
3. Check that unrelated provider results were rejected by news relevance validation.
4. Add a regression fixture before changing behavior.
5. Re-run the backend suite and deploy the smallest corrective change.

## CI or Docker failure

1. Identify the failing job and exact command.
2. Reproduce locally with the locked environment.
3. Check Dockerfile syntax, dependency installation, and healthcheck behavior.
4. Re-run tests and the complete image build after the fix.
