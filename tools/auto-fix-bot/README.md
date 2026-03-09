# 🤖 SageMaker Python SDK Auto Bug-Fix Bot

Automated bug-fix pipeline for [aws/sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk).

When a bug is filed on the repo (labeled `type: bug`), this bot:
1. **Parses** the issue (component, error, reproduction steps)
2. **Analyzes** the codebase to find relevant files and existing patterns
3. **Generates** a minimal fix + unit tests using an AI agent (Claude)
4. **Creates a PR** with a structured description, linked to the original issue

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────┐
│ GitHub Issue │────▶│ GitHub Actions   │────▶│  AI Agent        │────▶│ Create PR  │
│ (bug label) │     │ Workflow Trigger  │     │  (Claude API +   │     │ via `gh`   │
│             │     │                  │     │   tool-use loop) │     │            │
└─────────────┘     └──────────────────┘     └─────────────────┘     └────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- GitHub CLI (`gh`) authenticated
- `ANTHROPIC_API_KEY` or `AWS_BEDROCK_*` credentials
- Write access to the target repo (or a fork for testing)

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Test with a specific issue (dry-run, no PR created)
python -m bot.main --issue 5524 --repo aws/sagemaker-python-sdk --dry-run

# Full run (creates PR)
python -m bot.main --issue 5524 --repo aws/sagemaker-python-sdk
```

### GitHub Actions (Automated)
1. Copy `.github/workflows/auto-bug-fix.yml` to your repo
2. Set repository secrets:
   - `ANTHROPIC_API_KEY` (or Bedrock credentials)
   - `BOT_GITHUB_TOKEN` (PAT with repo write access)
3. Label an issue with `type: bug` → bot triggers automatically

## Project Structure

```
sagemaker-python-sdk-bot/
├── .github/
│   └── workflows/
│       └── auto-bug-fix.yml          # GitHub Actions workflow
├── bot/
│   ├── __init__.py
│   ├── main.py                       # CLI entry point & orchestrator
│   ├── issue_parser.py               # Parse GitHub issue into structured data
│   ├── codebase_explorer.py          # Clone repo, find relevant files
│   ├── agent.py                      # AI agent with tool-use loop
│   ├── tools.py                      # Tools the agent can use (read/write/search/run)
│   ├── pr_creator.py                 # Create branch, commit, open PR
│   ├── guardrails.py                 # Safety checks (max files, lines, test pass)
│   └── prompts/
│       ├── system.md                 # System prompt for the agent
│       ├── fix_bug.md                # Bug-fix task prompt template
│       └── pr_description.md         # PR description generation template
├── config/
│   └── config.yaml                   # Bot configuration
├── tests/
│   ├── test_issue_parser.py
│   ├── test_guardrails.py
│   └── test_agent.py
├── requirements.txt
└── README.md
```

## Safety Guardrails

| Guardrail | Default | Purpose |
|-----------|---------|---------|
| Max files changed | 5 | Prevent runaway edits |
| Max lines changed | 500 | Keep fixes focused |
| Tests must pass | ✅ | Don't create broken PRs |
| Human review required | ✅ | PR is never auto-merged |
| `auto-generated` label | ✅ | Clear attribution |
| Confidence threshold | 0.7 | Low confidence → draft PR |
| Rate limit | 5/day | Prevent spam |
| Component allowlist | configurable | Start narrow, expand |

## How It Works (Detailed)

### Step 1: Issue Parsing
Extracts structured data from the GitHub issue:
- **Component** (from labels like `component: pipelines`)
- **Error message** (from code blocks / stack traces)
- **Reproduction steps** (from "To reproduce" section)
- **Affected class/module** (from mentions in title/body)
- **SDK version** (V2 vs V3)

### Step 2: Codebase Exploration
- Clones the repo at `master`
- Uses file search + grep to find relevant source files
- Identifies existing patterns (how similar issues were solved)
- Maps dependencies and test files

### Step 3: AI Agent Fix
The agent operates in a tool-use loop:
1. Read relevant source files
2. Search for patterns across the codebase
3. Write the fix (minimal changes)
4. Write/update unit tests
5. Run tests to validate
6. Self-review the changes

### Step 4: PR Creation
- Branch: `auto-fix/issue-{number}`
- Commit: `fix: {description} (fixes #{number})`
- PR body: Structured (Summary, Problem, Solution, Testing)
- Labels: `auto-generated` + original bug labels
- Comment on original issue linking to PR

## Configuration

See `config/config.yaml` for all options.

## Origin Story

This bot was inspired by the manual fix of [#5524](https://github.com/aws/sagemaker-python-sdk/issues/5524) → [PR #5608](https://github.com/aws/sagemaker-python-sdk/pull/5608), where Cline was used to analyze the issue, explore the codebase, find existing patterns (`StrPipeVar`), make the fix, add tests, and create the PR. The entire workflow was automatable.
