# Copilot / AI agent instructions for this repository

Purpose: give an AI coding assistant the concrete, discoverable facts it needs to be immediately productive in this codebase.

Summary of repo state (discoverable):
- The repository currently contains a single file at the project root: `Readme.md` (note the capitalized filename).
- No build config files, dependency manifests, source folders (`src/`, `notebooks/`, `tests/`) or CI configs were found during an initial scan.

Actionable checklist for an AI agent (how to start):
1. Run a repository-wide scan for common files to learn the project's language and workflows. Look for these files/dirs and summarize their contents if present: `requirements.txt`, `pyproject.toml`, `setup.py`, `environment.yml`, `package.json`, `Pipfile`, `Dockerfile`, `Makefile`, `src/`, `notebooks/`, `tests/`, `data/`, `scripts/`, `models/`, `.github/`.
   - Example: "I found `pyproject.toml` declaring Python 3.11 and dependencies X,Y,Z; tests live under `tests/` and use pytest."
2. If no manifest/build/test files are present (as in the current snapshot), explicitly ask the human owner these questions before making major changes:
   - What language/runtime should I assume? (Python, Node, other)
   - How should I run the project locally? (example run command)
   - Any preferred dependency/virtualenv workflow? (venv/conda/poetry)

Big-picture guidance (what to look for and why):
- Files named `Readme.md` or `README.md` may contain essential intent. Open and summarize them early.
- Prioritize discovering a dependency manifest or a Dockerfile — they define the canonical runtime and test commands.
- If you find tests, run the test suite and add minimal passing tests for any changes. If no tests exist, add a focused unit test for the new behavior you introduce.

Project-specific patterns discovered (only documentable facts):
- The repo currently has minimal structure (single `Readme.md`). There are no project-specific coding conventions or CI workflows to rely on.
- Because the repo is empty/early-stage, prefer asking short clarifying questions rather than making large structural changes without approval.

How to produce a useful change/PR in this repo (practical examples):
- Discovery-first PR: A small PR that adds a `pyproject.toml`/`requirements.txt` or `README` expansion describing how to run and test. Keep changes minimal and well-documented.
- When adding new files, include a one-line purpose at the top of each new file and a matching test under `tests/`.

If you find an existing `.github/copilot-instructions.md` or AGENT.md when scanning, merge intelligently:
- Preserve any explicit instruction lines, example commands, or owner contacts.
- Add only non-redundant clarifications (what I scanned and why I added/changed the guidance).

Where to look for further context (priority order):
1. `Readme.md` at repo root
2. Any manifest files (`pyproject.toml`, `requirements.txt`, `package.json`, `Dockerfile`)
3. `src/`, `notebooks/`, `tests/`, `scripts/` directories

Finish by asking the user for missing facts. Use these exact prompts when information is missing:
- "Which runtime (language and version) should I assume for this repo?"
- "How do you run and test this project locally? Provide the exact commands you use."

If you (the developer) want me to iterate on this guidance, tell me what additional files or conventions exist in the project and I will merge them into this document.
