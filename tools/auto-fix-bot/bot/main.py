"""Main orchestrator for the auto bug-fix bot.

Refactored from a single 185-line main() function into an Orchestrator class
with testable steps and structured logging with trace IDs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import click
import yaml
from jinja2 import Template
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .config import BotConfig, load_validated_config
from .issue_parser import IssueParser, ParsedIssue
from .codebase_explorer import CodebaseExplorer
from .agent import BugFixAgent, AgentResult
from .guardrails import Guardrails, ChangeStats, compute_change_stats
from .pr_creator import PRCreator

console = Console()


# ──────────────────────────────────────────────
# Structured logging with trace IDs
# ──────────────────────────────────────────────

class TraceLogAdapter(logging.LoggerAdapter):
    """Adds run_id and issue context to every log message."""

    def process(self, msg, kwargs):
        extra = self.extra or {}
        prefix = f"[run={extra.get('run_id', '?')[:8]}|issue=#{extra.get('issue_number', '?')}]"
        return f"{prefix} {msg}", kwargs


def setup_logging(config: BotConfig, run_id: str, issue_number: int) -> TraceLogAdapter:
    """Set up logging with rich console output, file logging, and trace IDs."""
    level = getattr(logging, config.logging.level.upper(), logging.INFO)

    # Console handler (rich)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
        force=True,
    )

    # File handler with JSON-style structured lines
    log_dir = Path(config.logging.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "bot.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        f"%(asctime)s [%(levelname)s] run={run_id[:8]} issue=#{issue_number} %(name)s: %(message)s"
    ))
    logging.getLogger().addHandler(file_handler)

    return TraceLogAdapter(
        logging.getLogger("bot"),
        {"run_id": run_id, "issue_number": issue_number},
    )


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = Path(__file__).parent / "prompts" / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


# ──────────────────────────────────────────────
# Orchestrator — each step is a testable method
# ──────────────────────────────────────────────

class AutoFixOrchestrator:
    """Orchestrates the full bug-fix pipeline with testable steps."""

    def __init__(self, config: BotConfig, github_token: str | None, dry_run: bool = False):
        self.config = config
        self.github_token = github_token
        self.dry_run = dry_run
        self.run_id = str(uuid.uuid4())

    def run(self, issue_number: int, repo: str, component: str = "unknown") -> bool:
        """Execute the full pipeline. Returns True on success."""
        logger = setup_logging(self.config, self.run_id, issue_number)

        console.print(Panel(
            f"[bold blue]Auto Bug-Fix Bot[/bold blue]\n"
            f"Run: {self.run_id[:8]} | Issue: #{issue_number} | Repo: {repo}\n"
            f"Component: {component} | Dry run: {self.dry_run}",
            title="🤖 Starting",
        ))

        # Step 1: Parse issue
        parsed_issue = self.step_parse_issue(logger, repo, issue_number, component)
        if not parsed_issue:
            return False

        # Step 2: Pre-flight guardrail checks
        guardrails = Guardrails(self.config.guardrails.model_dump())
        if not self.step_preflight_checks(logger, guardrails, parsed_issue):
            return False

        # Step 3: Clone repo
        explorer, repo_path = self.step_clone_repo(logger, repo)
        if not repo_path:
            return False

        # Step 4: Run AI agent
        agent_result = self.step_run_agent(logger, parsed_issue, repo_path, explorer)
        if not agent_result or not agent_result.success:
            return False

        # Step 5: Post-fix guardrail checks
        change_stats, passed = self.step_guardrail_checks(
            logger, guardrails, repo_path, agent_result, parsed_issue
        )
        if not passed:
            return False

        # Step 6: Create PR (or dry-run report)
        return self.step_create_pr(
            logger, repo, repo_path, parsed_issue, agent_result, guardrails, change_stats
        )

    # ── Individual steps ──

    def step_parse_issue(
        self, logger: TraceLogAdapter, repo: str, issue_number: int, component: str
    ) -> ParsedIssue | None:
        """Step 1: Parse the GitHub issue."""
        console.print("\n[bold]Step 1: Parsing issue...[/bold]")
        parser = IssueParser(github_token=self.github_token)

        try:
            parsed = parser.parse_from_api(repo, issue_number)
        except Exception as e:
            logger.error("Failed to parse issue: %s", e)
            console.print(f"[red]Failed to fetch/parse issue #{issue_number}: {e}[/red]")
            return None

        if component != "unknown":
            parsed.component = component

        _print_parsed_issue(parsed)
        logger.info(
            "Parsed: component=%s, classes=%s, error=%s",
            parsed.component, parsed.affected_classes, parsed.error_type,
        )
        return parsed

    def step_preflight_checks(
        self, logger: TraceLogAdapter, guardrails: Guardrails, issue: ParsedIssue
    ) -> bool:
        """Step 2: Pre-flight guardrail checks."""
        console.print("\n[bold]Step 2: Pre-flight checks...[/bold]")

        label_check = guardrails.check_labels(issue.labels)
        if not label_check.passed:
            logger.warning("Blocked by label check: %s", label_check.message)
            console.print(f"[red]Blocked: {label_check.message}[/red]")
            return False

        component_check = guardrails.check_component(issue.component)
        if not component_check.passed:
            logger.warning("Component check: %s", component_check.message)
            console.print(f"[yellow]Warning: {component_check.message}[/yellow]")

        console.print("[green]Pre-flight checks passed[/green]")
        return True

    def step_clone_repo(
        self, logger: TraceLogAdapter, repo: str
    ) -> tuple[CodebaseExplorer, Path | None]:
        """Step 3: Clone the repository."""
        console.print("\n[bold]Step 3: Cloning repository...[/bold]")

        explorer = CodebaseExplorer(
            repo=repo,
            workspace_dir=self.config.workspace.dir,
            branch=self.config.repo.default_branch,
        )

        try:
            repo_path = explorer.clone()
            console.print(f"[green]Cloned to {repo_path}[/green]")
            logger.info("Cloned repo to %s", repo_path)
            return explorer, repo_path
        except Exception as e:
            logger.error("Failed to clone: %s", e)
            console.print(f"[red]Clone failed: {e}[/red]")
            return explorer, None

    def step_run_agent(
        self,
        logger: TraceLogAdapter,
        issue: ParsedIssue,
        repo_path: Path,
        explorer: CodebaseExplorer,
    ) -> AgentResult | None:
        """Step 4: Run the AI agent to fix the bug."""
        console.print("\n[bold]Step 4: Running AI agent...[/bold]")

        # Find relevant files
        relevant = explorer.find_relevant_files(issue.affected_classes, issue.affected_modules)
        logger.info("Relevant files: %s", relevant)

        # Render prompts
        system_prompt = load_prompt("system.md")
        task_template = Template(load_prompt("fix_bug.md"))
        task_prompt = task_template.render(
            issue_number=issue.number,
            issue_title=issue.title,
            issue_body=issue.body,
            component=issue.component,
            labels=", ".join(issue.labels),
            repo_path=str(repo_path),
        )

        # Run agent
        agent = BugFixAgent(self.config.ai.model_dump())
        result = agent.fix_bug(issue, repo_path, system_prompt, task_prompt)

        # Log token usage
        logger.info(
            "Agent done: success=%s, confidence=%.2f, iterations=%d, "
            "tokens=%d (in=%d, out=%d)",
            result.success, result.confidence, result.iterations,
            result.token_usage.total_tokens,
            result.token_usage.input_tokens,
            result.token_usage.output_tokens,
        )
        _print_agent_result(result)

        # Save conversation
        if self.config.logging.save_conversation:
            conv_path = Path(self.config.logging.dir) / f"conversation-{issue.number}.json"
            conv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(conv_path, "w") as f:
                json.dump(
                    _serialize_conversation(result.conversation_history),
                    f, indent=2, default=str,
                )
            logger.info("Saved conversation to %s", conv_path)

        if not result.success:
            console.print(f"[red]Agent could not fix the issue: {result.reason}[/red]")
            return None

        return result

    def step_guardrail_checks(
        self,
        logger: TraceLogAdapter,
        guardrails: Guardrails,
        repo_path: Path,
        result: AgentResult,
        issue: ParsedIssue,
    ) -> tuple[ChangeStats, bool]:
        """Step 5: Run guardrail checks on the changes."""
        console.print("\n[bold]Step 5: Running guardrail checks...[/bold]")

        # Stage changes to compute stats
        subprocess.run(["git", "add", "-A"], cwd=repo_path, capture_output=True, check=True)
        change_stats = compute_change_stats(repo_path)
        subprocess.run(["git", "reset", "HEAD"], cwd=repo_path, capture_output=True)

        guardrail_results = guardrails.check_all(
            repo_path=repo_path,
            change_stats=change_stats,
            confidence=result.confidence,
            component=issue.component,
            labels=issue.labels,
        )
        _print_guardrail_results(guardrail_results)

        if not guardrails.all_passed(guardrail_results):
            logger.error("Guardrail checks failed")
            console.print("[red]Guardrail checks failed — not creating PR[/red]")
            return change_stats, False

        return change_stats, True

    def step_create_pr(
        self,
        logger: TraceLogAdapter,
        repo: str,
        repo_path: Path,
        issue: ParsedIssue,
        result: AgentResult,
        guardrails: Guardrails,
        change_stats: ChangeStats,
    ) -> bool:
        """Step 6: Create the PR (or dry-run report)."""
        is_draft = guardrails.should_create_draft(result.confidence)

        if self.dry_run:
            console.print(Panel(
                f"[yellow]DRY RUN — would create PR:[/yellow]\n"
                f"Branch: auto-fix/issue-{issue.number}\n"
                f"Files: {', '.join(result.files_changed)}\n"
                f"Draft: {is_draft}\n"
                f"Confidence: {result.confidence:.2f}\n"
                f"Tokens: {result.token_usage.total_tokens:,}",
                title="🏁 Dry Run Result",
            ))
            return True

        console.print("\n[bold]Step 6: Creating PR...[/bold]")

        # Generate PR description
        pr_template = Template(load_prompt("pr_description.md"))
        pr_prompt = pr_template.render(
            issue_number=issue.number,
            issue_title=issue.title,
            component=issue.component,
            changes_summary=result.summary,
            files_changed=change_stats.files,
        )
        agent = BugFixAgent(self.config.ai.model_dump())
        pr_description = agent.generate_pr_description(issue, result, pr_prompt)

        # Create the PR
        pr_creator = PRCreator(self.config.pr.model_dump(), self.github_token)
        pr_url = pr_creator.create_pr(
            repo_name=repo,
            repo_path=repo_path,
            issue=issue,
            agent_result=result,
            pr_description=pr_description,
            is_draft=is_draft,
        )

        if pr_url:
            logger.info("PR created: %s (tokens=%d)", pr_url, result.token_usage.total_tokens)
            console.print(Panel(
                f"[green bold]PR created successfully![/green bold]\n\n"
                f"🔗 {pr_url}\n"
                f"📝 Confidence: {result.confidence:.0%}\n"
                f"📄 Files: {change_stats.files_changed}\n"
                f"➕ {change_stats.lines_added} additions, ➖ {change_stats.lines_deleted} deletions\n"
                f"🔢 Tokens: {result.token_usage.total_tokens:,}",
                title="🎉 Success",
            ))
            return True
        else:
            logger.error("Failed to create PR")
            console.print("[red]Failed to create PR[/red]")
            return False


# ──────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────

def _print_parsed_issue(issue: ParsedIssue) -> None:
    table = Table(title=f"Issue #{issue.number}")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Title", issue.title)
    table.add_row("Component", issue.component)
    table.add_row("SDK Version", issue.sdk_version)
    table.add_row("Labels", ", ".join(issue.labels))
    table.add_row("Error Type", issue.error_type or "(none)")
    table.add_row("Affected Classes", ", ".join(issue.affected_classes) or "(none)")
    table.add_row("Has Repro Code", "✅" if issue.reproduction_code else "❌")
    console.print(table)


def _print_agent_result(result: AgentResult) -> None:
    status = "[green]✅ Success[/green]" if result.success else "[red]❌ Failed[/red]"
    console.print(Panel(
        f"Status: {status}\n"
        f"Confidence: {result.confidence:.2f}\n"
        f"Iterations: {result.iterations}\n"
        f"Files: {', '.join(result.files_changed)}\n"
        f"Tokens: {result.token_usage.total_tokens:,}\n"
        f"Summary: {result.summary}",
        title="Agent Result",
    ))


def _print_guardrail_results(results: list) -> None:
    table = Table(title="Guardrail Checks")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details")
    for r in results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        label = r.message.split(":")[0] if ":" in r.message else "Check"
        table.add_row(label, status, r.message)
    console.print(table)


def _serialize_conversation(conversation: list[dict]) -> list[dict]:
    serialized = []
    for msg in conversation:
        entry = {"role": msg["role"]}
        content = msg.get("content")
        if isinstance(content, str):
            entry["content"] = content
        elif isinstance(content, list):
            entry["content"] = []
            for item in content:
                if hasattr(item, "type"):
                    if item.type == "text":
                        entry["content"].append({"type": "text", "text": item.text})
                    elif item.type == "tool_use":
                        entry["content"].append({"type": "tool_use", "name": item.name, "input": item.input})
                elif isinstance(item, dict):
                    entry["content"].append(item)
        serialized.append(entry)
    return serialized


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

@click.command()
@click.option("--issue", required=True, type=int, help="GitHub issue number to fix")
@click.option("--repo", required=True, help="Repository (owner/name)")
@click.option("--component", default="unknown", help="Component name (from triage)")
@click.option("--config-file", default="config/config.yaml", help="Config file path")
@click.option("--dry-run", is_flag=True, help="Don't create PR, just show what would happen")
def main(issue: int, repo: str, component: str, config_file: str, dry_run: bool):
    """Auto-fix a GitHub issue by analyzing and patching the code."""
    # Load and validate config
    try:
        config = load_validated_config(config_file)
    except Exception as e:
        console.print(f"[red]Config error: {e}[/red]")
        sys.exit(1)

    github_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not github_token and not dry_run:
        console.print("[red]Error: GH_TOKEN or GITHUB_TOKEN environment variable required[/red]")
        sys.exit(1)

    # Run the pipeline
    orchestrator = AutoFixOrchestrator(config, github_token, dry_run)
    success = orchestrator.run(issue, repo, component)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
