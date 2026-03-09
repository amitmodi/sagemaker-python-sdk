"""Create Git branches, commits, and Pull Requests."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

from github import Github

from .issue_parser import ParsedIssue
from .agent import AgentResult
from .guardrails import ChangeStats

logger = logging.getLogger(__name__)


class PRCreator:
    """Handles git operations and PR creation."""

    def __init__(self, config: dict[str, Any], github_token: str):
        """Initialize the PR creator.

        Args:
            config: The 'pr' section of config.yaml
            github_token: GitHub token with repo write access
        """
        self.branch_template = config.get(
            "branch_template", "auto-fix/issue-{issue_number}"
        )
        self.commit_template = config.get(
            "commit_template", "fix: {short_description} (fixes #{issue_number})"
        )
        self.labels = config.get("labels", ["auto-generated"])
        self.draft_on_low_confidence = config.get("draft_on_low_confidence", True)
        self.comment_on_issue = config.get("comment_on_issue", True)
        self.github_token = github_token
        self._github = Github(github_token)

    def create_pr(
        self,
        repo_name: str,
        repo_path: Path,
        issue: ParsedIssue,
        agent_result: AgentResult,
        pr_description: str,
        is_draft: bool = False,
    ) -> Optional[str]:
        """Create a branch, commit changes, push, and open a PR.

        Args:
            repo_name: Repository in "owner/name" format
            repo_path: Local path to the cloned repo
            issue: The parsed issue
            agent_result: Result from the agent
            pr_description: Markdown PR body
            is_draft: Whether to create as a draft PR

        Returns:
            URL of the created PR, or None if failed
        """
        branch_name = self.branch_template.format(issue_number=issue.number)
        commit_msg = self.commit_template.format(
            short_description=agent_result.summary[:60],
            issue_number=issue.number,
        )

        try:
            # 1. Create branch
            self._git(repo_path, "checkout", "-b", branch_name)
            logger.info("Created branch: %s", branch_name)

            # 2. Stage all changes
            self._git(repo_path, "add", "-A")

            # 3. Configure git user for the commit
            self._git(
                repo_path, "config", "user.email", "auto-fix-bot@github.com"
            )
            self._git(repo_path, "config", "user.name", "Auto-Fix Bot")

            # 4. Commit
            self._git(repo_path, "commit", "-m", commit_msg)
            logger.info("Committed: %s", commit_msg)

            # 5. Push branch (use credential helper to avoid token in process args)
            self._git(
                repo_path, "config", "credential.helper",
                f"!f() {{ echo username=x-access-token; echo password={self.github_token}; }}; f",
            )
            self._git(
                repo_path, "remote", "set-url", "origin",
                f"https://github.com/{repo_name}.git",
            )
            self._git(
                repo_path, "push", "origin", f"HEAD:{branch_name}"
            )
            logger.info("Pushed branch: %s", branch_name)

            # 6. Create the PR via GitHub API
            gh_repo = self._github.get_repo(repo_name)
            pr_title = self.commit_template.format(
                short_description=issue.short_description,
                issue_number=issue.number,
            )

            pr = gh_repo.create_pull(
                title=pr_title,
                body=pr_description,
                head=branch_name,
                base=gh_repo.default_branch,
                draft=is_draft,
            )
            logger.info("Created PR: %s", pr.html_url)

            # 7. Add labels
            all_labels = self.labels + issue.labels
            try:
                pr.add_to_labels(*all_labels)
            except Exception as e:
                logger.warning("Could not add labels: %s", e)

            # 8. Comment on the original issue
            if self.comment_on_issue:
                self._comment_on_issue(
                    gh_repo, issue, pr.html_url, is_draft,
                    confidence=agent_result.confidence,
                )

            return pr.html_url

        except Exception as e:
            logger.error("Failed to create PR: %s", e)
            return None

    def _comment_on_issue(
        self, gh_repo, issue: ParsedIssue, pr_url: str, is_draft: bool,
        confidence: float = 0.0,
    ) -> None:
        """Add a comment on the original issue linking to the PR."""
        try:
            gh_issue = gh_repo.get_issue(issue.number)
            draft_note = " (draft — low confidence, needs review)" if is_draft else ""
            comment = (
                f"🤖 **Auto-Fix Bot** created a PR for this issue{draft_note}:\n\n"
                f"➡️ {pr_url}\n\n"
                f"**Agent confidence**: {confidence:.0%}\n\n"
                f"Please review the changes. This PR was generated automatically "
                f"and requires human approval before merging."
            )
            gh_issue.create_comment(comment)
        except Exception as e:
            logger.warning("Could not comment on issue: %s", e)

    def _git(self, repo_path: Path, *args: str) -> str:
        """Run a git command in the repo directory.

        Returns:
            stdout of the command
        Raises:
            RuntimeError if the command fails
        """
        result = subprocess.run(
            ["git"] + list(args),
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {result.stderr}"
            )
        return result.stdout.strip()
