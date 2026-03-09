"""AI agent that analyzes and fixes bugs using Claude's tool-use capability.

This is the core of the bot — it runs a tool-use loop where Claude reads the
codebase, identifies the fix, makes changes, writes tests, and validates.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import anthropic

from .issue_parser import ParsedIssue
from .tools import TOOL_DEFINITIONS, ToolExecutor

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Tracks cumulative token usage across all API calls."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AgentResult:
    """Result from the agent's bug-fix attempt."""

    success: bool
    confidence: float
    summary: str
    files_changed: list[str] = field(default_factory=list)
    reason: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    iterations: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)


# Retry configuration
MAX_API_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds
RETRY_MAX_DELAY = 30.0  # seconds


def _call_with_retry(fn, max_retries: int = MAX_API_RETRIES, **kwargs) -> Any:
    """Call a function with exponential backoff retry on transient errors.

    Retries on rate limits (429), server errors (5xx), and connection errors.
    Does NOT retry on auth errors (401/403) or bad request (400).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(**kwargs)
        except anthropic.RateLimitError as e:
            last_exc = e
            if attempt < max_retries:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries + 1, delay, e,
                )
                time.sleep(delay)
            else:
                raise
        except anthropic.InternalServerError as e:
            last_exc = e
            if attempt < max_retries:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                logger.warning(
                    "Server error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries + 1, delay, e,
                )
                time.sleep(delay)
            else:
                raise
        except anthropic.APIConnectionError as e:
            last_exc = e
            if attempt < max_retries:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                logger.warning(
                    "Connection error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries + 1, delay, e,
                )
                time.sleep(delay)
            else:
                raise
    # Should not reach here, but just in case
    raise last_exc  # type: ignore[misc]


class BugFixAgent:
    """AI agent that fixes bugs using Claude's tool-use loop."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the agent.

        Args:
            config: The 'ai' section of config.yaml
        """
        # All values come from Pydantic-validated BotConfig — no hardcoded defaults here
        self.provider = config["provider"]
        self.model = config["model"]
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
        self.max_iterations = config["max_iterations"]

        # Initialize the appropriate client
        if self.provider == "bedrock":
            bedrock_config = config.get("bedrock", {})
            self.client = anthropic.AnthropicBedrock(
                aws_region=bedrock_config["region"],
            )
            self.model = bedrock_config["model_id"]
        else:
            self.client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )

    def fix_bug(
        self,
        issue: ParsedIssue,
        repo_path: Path,
        system_prompt: str,
        task_prompt: str,
    ) -> AgentResult:
        """Run the agent to fix a bug.

        This is the main entry point — it runs a tool-use loop until the agent
        calls report_result or hits the max iterations.

        Args:
            issue: Parsed issue data
            repo_path: Path to the cloned repository
            system_prompt: System prompt (from prompts/system.md)
            task_prompt: Task prompt (rendered from prompts/fix_bug.md)

        Returns:
            AgentResult with the outcome
        """
        tool_executor = ToolExecutor(repo_path)
        conversation: list[dict] = []
        result = AgentResult(
            success=False,
            confidence=0.0,
            summary="Agent did not complete",
        )

        # Start with the task prompt as the first user message
        conversation.append({
            "role": "user",
            "content": task_prompt,
        })

        logger.info("Starting bug-fix agent for issue #%d", issue.number)

        for iteration in range(self.max_iterations):
            result.iterations = iteration + 1
            logger.info("Agent iteration %d/%d", iteration + 1, self.max_iterations)

            try:
                # Call Claude with retry on transient errors
                response = _call_with_retry(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    tools=TOOL_DEFINITIONS,
                    messages=conversation,
                )
            except Exception as e:
                logger.error("API call failed after retries: %s", e)
                result.reason = f"API error: {e}"
                break

            # Track token usage
            if hasattr(response, "usage") and response.usage:
                result.token_usage.input_tokens += response.usage.input_tokens
                result.token_usage.output_tokens += response.usage.output_tokens
                logger.info(
                    "Tokens this call: in=%d out=%d | Cumulative: in=%d out=%d | Total=%d",
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    result.token_usage.input_tokens,
                    result.token_usage.output_tokens,
                    result.token_usage.total_tokens,
                )

            # Process the response
            assistant_message = {"role": "assistant", "content": response.content}
            conversation.append(assistant_message)

            # Check if the agent wants to use tools
            if response.stop_reason == "tool_use":
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        logger.info("Tool call: %s(%s)", tool_name, json.dumps(tool_input)[:200])

                        # Check for the terminal tool
                        if tool_name == "report_result":
                            result.success = tool_input.get("success", False)
                            result.confidence = tool_input.get("confidence", 0.0)
                            result.summary = tool_input.get("summary", "")
                            result.files_changed = tool_input.get("files_changed", tool_executor.files_written)
                            result.reason = tool_input.get("reason", "")
                            result.conversation_history = conversation
                            logger.info(
                                "Agent reported result: success=%s, confidence=%.2f",
                                result.success,
                                result.confidence,
                            )
                            return result

                        # Execute the tool
                        tool_result = tool_executor.execute(tool_name, tool_input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result["content"],
                            "is_error": tool_result.get("is_error", False),
                        })

                # Add tool results to conversation
                conversation.append({
                    "role": "user",
                    "content": tool_results,
                })

            elif response.stop_reason == "end_turn":
                # Agent finished without calling report_result
                # Extract any text content as the summary
                text_blocks = [b.text for b in response.content if hasattr(b, "text")]
                if text_blocks:
                    result.summary = "\n".join(text_blocks)
                result.files_changed = tool_executor.files_written
                result.conversation_history = conversation
                logger.warning("Agent ended without calling report_result")
                break

            else:
                logger.warning("Unexpected stop reason: %s", response.stop_reason)
                break

        # If we hit max iterations
        if result.iterations >= self.max_iterations:
            result.reason = f"Hit max iterations ({self.max_iterations})"
            logger.warning("Agent hit max iterations")

        result.conversation_history = conversation
        return result

    def generate_pr_description(
        self,
        issue: ParsedIssue,
        agent_result: AgentResult,
        pr_template: str,
    ) -> str:
        """Generate a PR description using Claude.

        Args:
            issue: The parsed issue
            agent_result: Result from the fix attempt
            pr_template: The PR description prompt template (already rendered)

        Returns:
            Markdown-formatted PR description
        """
        try:
            response = _call_with_retry(
                self.client.messages.create,
                model=self.model,
                max_tokens=2048,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": pr_template,
                }],
            )

            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            return "\n".join(text_blocks)

        except Exception as e:
            logger.error("Failed to generate PR description after retries: %s", e)
            # Fallback to a basic description
            return (
                f"## Summary\n{agent_result.summary}\n\n"
                f"**Fixes**: #{issue.number}\n\n"
                f"## Files Changed\n"
                + "\n".join(f"- `{f}`" for f in agent_result.files_changed)
                + "\n\n*Auto-generated by bug-fix bot*"
            )
