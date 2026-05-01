# Contributing

Thanks for your interest in contributing. Please read this before opening a PR.

## Before you start

**Open an issue first.** Check existing issues and discussions — there may already be a planned or in-progress implementation. A quick issue saves everyone time and avoids duplicate work. Issues are cheap; rebased PRs are not.

## AI-generated code

Code written or substantially assisted by an AI agent (Copilot, Claude, Cursor, etc.) must be reviewed and vouched for by a human contributor before it can be merged. This means:

- You have read and understand every line of the proposed change.
- You are personally accountable for its correctness, security, and fit with the codebase.
- The PR description must disclose that AI tooling was used.

We welcome agent-assisted contributions — we just need a human in the loop who can answer questions and own the code after it lands.

## General expectations

- Be respectful. Review comments are about the code, not the person.
- Keep PRs focused. One logical change per PR makes review faster and history cleaner.
- Write tests for new behavior. See `tests/` and the harness notes in `CLAUDE.md`.
- Run the full pre-commit suite before pushing (`SKIP=pytest pre-commit run --all-files`).
- If your change touches the compile pipeline or canary outputs, regenerate the golden files.

## Pull request checklist

- [ ] Issue exists and links to this PR
- [ ] AI tooling disclosed in PR description (if applicable)
- [ ] Tests added or updated
- [ ] Pre-commit passes locally
- [ ] CLAUDE.md updated if architecture or commands changed
