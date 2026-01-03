from src.tools import tool

import os
import shlex
from pydantic import Field
from collections.abc import Sequence
from openhands.sdk import (
    Action,
    ImageContent,
    Observation,
    TextContent,
    ToolDefinition,
)
from openhands.sdk.tool import (
    ToolExecutor,
)
from openhands.tools.terminal import (
    TerminalAction,
    TerminalExecutor,
    TerminalTool,
)

# --- Action / Observation ---


class CustomGrepAction(Action):
    pattern: str = Field(description="Regex to search for")
    path: str = Field(
        default=".", description="Directory to search (absolute or relative)"
    )
    include: str | None = Field(
        default=None, description="Optional glob to filter files (e.g. '*.py')"
    )

class CustomGrepObservation(Observation):
    matches: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    count: int = 0

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if not self.count:
            return [TextContent(text="No matches found.")]
        files_list = "\n".join(f"- {f}" for f in self.files[:20])
        sample = "\n".join(self.matches[:10])
        more = "\n..." if self.count > 10 else ""
        ret = (
            f"Found {self.count} matching lines.\n"
            f"Files:\n{files_list}\n"
            f"Sample:\n{sample}{more}"
        )
        return [TextContent(text=ret)]

# --- Executor ---


class CustomGrepExecutor(ToolExecutor[CustomGrepAction, CustomGrepObservation]):
    def __init__(self, terminal: TerminalExecutor):
        self.terminal: TerminalExecutor = terminal

    def __call__(self, action: CustomGrepAction, conversation=None) -> CustomGrepObservation:  # noqa: ARG002
        root = os.path.abspath(action.path)
        pat = shlex.quote(action.pattern)
        root_q = shlex.quote(root)

        # Use grep -r; add --include when provided
        if action.include:
            inc = shlex.quote(action.include)
            cmd = f"grep -rHnE --include {inc} {pat} {root_q} 2>/dev/null | head -100"
        else:
            cmd = f"grep -rHnE {pat} {root_q} 2>/dev/null | head -100"

        result = self.terminal(TerminalAction(command=cmd))

        matches: list[str] = []
        files: set[str] = set()

        # grep returns exit code 1 when no matches; treat as empty
        output_text = result.text

        if output_text.strip():
            for line in output_text.strip().splitlines():
                matches.append(line)
                # Expect "path:line:content" — take the file part before first ":"
                file_path = line.split(":", 1)[0]
                if file_path:
                    files.add(os.path.abspath(file_path))

        return CustomGrepObservation(matches=matches, files=sorted(files), count=len(matches))

# Tool description
_GREP_DESCRIPTION = """Fast content search tool.
* Searches file contents using regular expressions
* Supports full regex syntax (eg. "log.*Error", "function\\s+\\w+", etc.)
* Filter files by pattern with the include parameter (eg. "*.js", "*.{ts,tsx}")
* Returns matching file paths sorted by modification time.
* Only the first 100 results are returned. Consider narrowing your search with stricter regex patterns or provide path parameter if you need more results.
* Use this tool when you need to find files containing specific patterns
* When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead
"""  # noqa: E501

# --- Tool Definition ---

class CustomGrepTool(ToolDefinition[CustomGrepAction, CustomGrepObservation]):
    """A custom grep tool that searches file contents using regular expressions."""

    @classmethod
    def create(
        cls, conv_state, terminal_executor: TerminalExecutor | None = None
    ) -> Sequence[ToolDefinition]:
        """Create CustomGrepTool instance with a CustomGrepExecutor.

        Args:
            conv_state: Conversation state to get working directory from.
            terminal_executor: Optional terminal executor to reuse. If not provided,
                         a new one will be created.

        Returns:
            A sequence containing a single CustomGrepTool instance.
        """
        if terminal_executor is None:
            terminal_executor = TerminalExecutor(
                working_dir=conv_state.workspace.working_dir
            )
        grep_executor = CustomGrepExecutor(terminal_executor)

        return [
            cls(
                description=_GREP_DESCRIPTION,
                action_type=CustomGrepAction,
                observation_type=CustomGrepObservation,
                executor=grep_executor,
            )
        ]

@tool(name="bash_and_grep_toolset")
def _make_bash_and_grep_tools(conv_state) -> list[ToolDefinition]:
    """Create terminal and custom grep tools sharing one executor."""

    terminal_executor = TerminalExecutor(working_dir=conv_state.workspace.working_dir)
    # terminal_tool = terminal_tool.set_executor(executor=terminal_executor)
    terminal_tool = TerminalTool.create(conv_state, executor=terminal_executor)[0]

    # Use the CustomGrepTool.create() method with shared terminal_executor
    grep_tool = CustomGrepTool.create(conv_state, terminal_executor=terminal_executor)[0]

    return [terminal_tool, grep_tool]