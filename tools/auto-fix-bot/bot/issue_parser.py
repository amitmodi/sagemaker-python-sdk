"""Parse GitHub issues into structured data for the AI agent."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from github import Github
from github.Issue import Issue as GHIssue

logger = logging.getLogger(__name__)


@dataclass
class ParsedIssue:
    """Structured representation of a GitHub bug report."""

    number: int
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    component: str = "unknown"
    sdk_version: str = "v3"  # v2 or v3
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    reproduction_code: Optional[str] = None
    expected_behavior: Optional[str] = None
    affected_classes: list[str] = field(default_factory=list)
    affected_modules: list[str] = field(default_factory=list)
    url: str = ""

    @property
    def short_description(self) -> str:
        """Generate a short description suitable for commit messages."""
        # Remove common prefixes
        title = self.title
        for prefix in ["[Bug]", "[bug]", "Bug:", "bug:"]:
            title = title.replace(prefix, "").strip()
        # Truncate to 72 chars for git
        if len(title) > 72:
            title = title[:69] + "..."
        return title


class IssueParser:
    """Parses a GitHub issue into structured data."""

    # Patterns for extracting information
    CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
    ERROR_RE = re.compile(
        r"(?:Error|Exception|Traceback).*?(?=\n\n|\n```|\Z)", re.DOTALL
    )
    CLASS_RE = re.compile(r"`?([A-Z][a-zA-Z]+(?:Trainer|Estimator|Step|Session|Config|Model|Pipeline))`?")
    MODULE_RE = re.compile(r"(?:sagemaker\.[\w.]+)")

    # Section headers in issue templates
    SECTION_HEADERS = {
        "reproduce": re.compile(
            r"(?:to reproduce|reproduction|repro|steps to reproduce|how to reproduce)",
            re.IGNORECASE,
        ),
        "expected": re.compile(
            r"(?:expected behavior|expected result|should)",
            re.IGNORECASE,
        ),
        "error": re.compile(
            r"(?:error|exception|traceback|stack trace|actual behavior)",
            re.IGNORECASE,
        ),
    }

    def __init__(self, github_token: Optional[str] = None):
        """Initialize with optional GitHub token for API access."""
        self._github = Github(github_token) if github_token else None

    def parse_from_api(self, repo: str, issue_number: int) -> ParsedIssue:
        """Fetch and parse an issue from the GitHub API.

        Args:
            repo: Repository in "owner/name" format (e.g., "aws/sagemaker-python-sdk")
            issue_number: The issue number

        Returns:
            ParsedIssue with structured data
        """
        if not self._github:
            raise ValueError("GitHub token required for API access")

        gh_repo = self._github.get_repo(repo)
        issue = gh_repo.get_issue(issue_number)
        return self.parse(issue)

    def parse(self, issue: GHIssue) -> ParsedIssue:
        """Parse a PyGithub Issue object into structured data."""
        return self.parse_raw(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            labels=[label.name for label in issue.labels],
            url=issue.html_url,
        )

    def parse_raw(
        self,
        number: int,
        title: str,
        body: str,
        labels: list[str] | None = None,
        url: str = "",
    ) -> ParsedIssue:
        """Parse raw issue data into structured form.

        This is the core parsing method — works without GitHub API.
        """
        labels = labels or []

        parsed = ParsedIssue(
            number=number,
            title=title,
            body=body,
            labels=labels,
            url=url,
        )

        # Extract component from labels
        parsed.component = self._extract_component(labels)

        # Detect SDK version
        parsed.sdk_version = self._detect_sdk_version(body)

        # Extract code blocks
        code_blocks = self.CODE_BLOCK_RE.findall(body)

        # Extract reproduction code (first code block, or the one after "reproduce" header)
        parsed.reproduction_code = self._extract_reproduction(body, code_blocks)

        # Extract error message
        parsed.error_message, parsed.error_type = self._extract_error(body, code_blocks)

        # Extract expected behavior
        parsed.expected_behavior = self._extract_section(body, "expected")

        # Extract affected classes and modules
        parsed.affected_classes = self._extract_classes(title + "\n" + body)
        parsed.affected_modules = self._extract_modules(body)

        logger.info(
            "Parsed issue #%d: component=%s, classes=%s, error_type=%s",
            number,
            parsed.component,
            parsed.affected_classes,
            parsed.error_type,
        )

        return parsed

    def _extract_component(self, labels: list[str]) -> str:
        """Extract component from labels like 'component: pipelines'."""
        for label in labels:
            if label.startswith("component:"):
                return label.split(":", 1)[1].strip()
        return "unknown"

    def _detect_sdk_version(self, body: str) -> str:
        """Detect which SDK version the issue is about."""
        # Check checkboxes in issue template
        if re.search(r"\[x\].*?V3|v3|3\.x", body):
            return "v3"
        if re.search(r"\[x\].*?V2|v2|2\.x", body):
            return "v2"
        # Default to v3
        return "v3"

    def _extract_reproduction(
        self, body: str, code_blocks: list[str]
    ) -> Optional[str]:
        """Extract reproduction code from the issue body."""
        # Try to find code after "reproduce" section header
        section = self._extract_section(body, "reproduce")
        if section:
            section_blocks = self.CODE_BLOCK_RE.findall(section)
            if section_blocks:
                return section_blocks[0].strip()

        # Fall back to first code block that looks like Python
        for block in code_blocks:
            block = block.strip()
            if any(
                keyword in block
                for keyword in ["import ", "from ", "def ", "class ", "="]
            ):
                return block

        return None

    def _extract_error(
        self, body: str, code_blocks: list[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """Extract error message and type from the issue."""
        # Check in code blocks first
        for block in code_blocks:
            # Look for common error patterns
            error_match = re.search(
                r"((?:ValidationError|TypeError|ImportError|AttributeError|"
                r"ValueError|RuntimeError|KeyError|ModuleNotFoundError)"
                r".*?)$",
                block,
                re.MULTILINE,
            )
            if error_match:
                error_msg = error_match.group(1).strip()
                error_type = error_msg.split(":")[0].split("(")[0].strip()
                return error_msg, error_type

            # Look for Traceback
            if "Traceback" in block:
                lines = block.strip().split("\n")
                if lines:
                    last_line = lines[-1].strip()
                    error_type = last_line.split(":")[0].strip()
                    return last_line, error_type

        # Check in body text
        error_section = self._extract_section(body, "error")
        if error_section:
            blocks_in_section = self.CODE_BLOCK_RE.findall(error_section)
            if blocks_in_section:
                return blocks_in_section[0].strip(), None

        return None, None

    def _extract_section(self, body: str, section_key: str) -> Optional[str]:
        """Extract a section from the issue body by header pattern."""
        pattern = self.SECTION_HEADERS.get(section_key)
        if not pattern:
            return None

        # Find the section header
        lines = body.split("\n")
        start_idx = None
        for i, line in enumerate(lines):
            if pattern.search(line):
                start_idx = i + 1
                break

        if start_idx is None:
            return None

        # Collect content until next header or end
        section_lines = []
        for line in lines[start_idx:]:
            # Stop at next markdown header
            if re.match(r"^#{1,3}\s", line) or re.match(r"^\*\*\w", line):
                break
            section_lines.append(line)

        return "\n".join(section_lines).strip() or None

    def _extract_classes(self, text: str) -> list[str]:
        """Extract class names mentioned in the text."""
        matches = self.CLASS_RE.findall(text)
        # Deduplicate while preserving order
        seen = set()
        result = []
        for cls in matches:
            if cls not in seen:
                seen.add(cls)
                result.append(cls)
        return result

    def _extract_modules(self, text: str) -> list[str]:
        """Extract sagemaker module paths mentioned in the text."""
        matches = self.MODULE_RE.findall(text)
        seen = set()
        result = []
        for mod in matches:
            if mod not in seen:
                seen.add(mod)
                result.append(mod)
        return result
