---
name: Post-merge setup
description: Durable setup-hook behavior for task merges in this workspace.
---

Post-merge setup runs the configured script from the project root with bash and closed stdin. The script must be idempotent, non-interactive, fail fast, and short enough for the configured timeout.

**Why:** A merged task failed before running any setup because `.replit` had no `[postMerge]` path configured.

**How to apply:** Keep dependency installation deterministic with lockfiles and avoid commands that prompt for input. Validate the hook by running the platform’s post-merge setup runner after configuring it.