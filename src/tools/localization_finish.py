"""Custom finish tool for code localization tasks.

This tool allows the agent to submit localization results in a structured format where:
- File path is required
- Class name is optional
- Function name is optional
"""

import os
from typing import TYPE_CHECKING
from collections.abc import Sequence

from pydantic import BaseModel, Field, computed_field
from rich.text import Text

from openhands.sdk import (
    Action,
    Observation,
    ToolDefinition,
)
from openhands.sdk.tool import ToolExecutor

from src.tools import tool

if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation


class CodeLocation(BaseModel):
    """A single code location with optional class and function."""

    file: str = Field(description="Path to the file (required)")
    class_name: str | None = Field(default=None, description="Class name (optional)")
    function_name: str | None = Field(default=None, description="Function/method name (optional)")


class LocalizationFinishAction(Action):
    """Action for submitting final localization results."""

    locations: list[CodeLocation] = Field(
        description="""List of code locations to modify. Each location must have:
- file: Path to the file (required)
- class_name: Class name (optional, omit for file-level or standalone functions)
- function_name: Function/method name (optional, omit for file-level or class-level only)

Examples:
- File-level change: {"file": "src/config.py"}
- Class-level change: {"file": "src/user.py", "class_name": "User"}
- Standalone function: {"file": "src/utils.py", "function_name": "helper"}
- Method in class: {"file": "src/parser.py", "class_name": "Parser", "function_name": "parse"}
"""
    )

    @computed_field
    @property
    def message(self) -> str:
        """Auto-generate message from locations for backward compatibility."""
        if not self.locations:
            return ""

        lines = []
        for loc in self.locations:
            lines.append(loc.file)
            if loc.class_name:
                lines.append(f"class: {loc.class_name}")
            if loc.function_name:
                lines.append(f"function: {loc.function_name}")
            lines.append("")  # Empty line between locations

        return "```\n" + "\n".join(lines).rstrip() + "\n```"

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this action."""
        content = Text()
        content.append("Submitting localization results:\n", style="bold blue")
        content.append(f"Found {len(self.locations)} location(s):\n", style="green")
        for i, loc in enumerate(self.locations, 1):
            content.append(f"  {i}. {loc.file}", style="cyan")
            if loc.class_name:
                content.append(f" → {loc.class_name}", style="yellow")
            if loc.function_name:
                content.append(f".{loc.function_name}", style="magenta")
            content.append("\n")
        return content


class LocalizationFinishObservation(Observation):
    """Observation returned after submitting localization results."""

    success: bool = Field(default=True, description="Whether submission was successful")
    num_locations: int = Field(default=0, description="Number of locations submitted")
    validation_message: str = Field(default="", description="Validation feedback")
    details: dict = Field(default_factory=dict, description="Additional details")

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this observation."""
        content = Text()
        if self.success:
            content.append(f"✓ Successfully submitted {self.num_locations} location(s)\n", style="bold green")
        else:
            content.append(f"✗ Submission failed\n", style="bold red")
            content.append(f"{self.validation_message}\n", style="yellow")
        return content


def locations_to_dict_list(locations: list[CodeLocation]) -> list[dict]:
    """Convert CodeLocation objects to dictionary format.

    Args:
        locations: List of CodeLocation objects

    Returns:
        List of dictionaries with 'file', 'class', 'function' keys
    """
    return [
        {
            "file": loc.file,
            "class": loc.class_name,
            "function": loc.function_name,
        }
        for loc in locations
    ]


class LocalizationFinishExecutor(ToolExecutor):
    """Executor for localization finish tool with validation."""

    def __init__(self, workspace_dir: str | None = None):
        """Initialize the executor.

        Args:
            workspace_dir: Optional workspace directory to validate file existence.
        """
        self.workspace_dir = workspace_dir

    def __call__(
        self,
        action: LocalizationFinishAction,
        conversation: "BaseConversation | None" = None,
    ) -> LocalizationFinishObservation:
        """Execute the finish action with validation.

        Args:
            action: The localization finish action to execute
            conversation: Optional conversation context

        Returns:
            LocalizationFinishObservation with validation results
        """

        try:
            # Get locations from action (already structured)
            locations = action.locations
            num_locs = len(locations)

            # Validation 1: Check if any locations were provided
            if num_locs == 0:
                return LocalizationFinishObservation(
                    success=False,
                    num_locations=0,
                    validation_message=(
                        "No locations provided. Please provide at least one location."
                    ),
                    details={"error": "empty_output"}
                )

            # Validation 2: Check each location has a file path
            errors = []
            for i, loc in enumerate(locations, 1):
                if not loc.file:
                    errors.append(f"Location {i} is missing a file path")

            if errors:
                return LocalizationFinishObservation(
                    success=False,
                    num_locations=0,
                    validation_message="\n".join(errors),
                    details={"error": "missing_file_paths", "locations": locations_to_dict_list(locations)}
                )

            # Validation 3: Check file existence (if workspace provided)
            if self.workspace_dir:
                missing_files = []
                for loc in locations:
                    file_path = loc.file
                    full_path = os.path.join(self.workspace_dir, file_path)
                    if not os.path.exists(full_path):
                        missing_files.append(file_path)

                if missing_files:
                    return LocalizationFinishObservation(
                        success=False,
                        num_locations=num_locs,
                        validation_message=(
                            f"Warning: {len(missing_files)} file(s) not found in workspace:\n" +
                            "\n".join(f"  - {f}" for f in missing_files[:5]) +
                            (f"\n  ... and {len(missing_files) - 5} more" if len(missing_files) > 5 else "")
                        ),
                        details={
                            "warning": "files_not_found",
                            "missing_files": missing_files,
                            "locations": locations_to_dict_list(locations)
                        }
                    )

            # Success!
            return LocalizationFinishObservation(
                success=True,
                num_locations=num_locs,
                validation_message=f"Successfully submitted {num_locs} location(s).",
                details={"locations": locations_to_dict_list(locations)}
            )

        except Exception as e:
            # Validation failed
            return LocalizationFinishObservation(
                success=False,
                num_locations=0,
                validation_message=(
                    f"Error validating locations: {str(e)}\n\n"
                    "Please ensure each location has a valid file path."
                ),
                details={"error": "validation_error", "exception": str(e)}
            )


TOOL_DESCRIPTION = """Submit your final code localization results.

Use this tool when you have identified all relevant files, classes, and functions
that need to be modified to address the issue described in the problem statement.

Provide a structured list of locations. Each location must have:
- file: Path to the file (required)
- class_name: Class name (optional)
- function_name: Function/method name (optional)

Examples of different scenarios:

1. File-level change (imports, globals, new top-level classes):
   {"file": "src/config.py"}

2. Class-level change (new methods, class attributes, entire class modified):
   {"file": "src/models/user.py", "class_name": "User"}

3. Standalone function (top-level function, not in a class):
   {"file": "src/utils/helpers.py", "function_name": "format_date"}

4. Method in a class (specific method modification):
   {"file": "src/parser.py", "class_name": "DataParser", "function_name": "parse_json"}

The tool will validate file existence and provide feedback if issues are found.
"""


class LocalizationFinishTool(ToolDefinition[LocalizationFinishAction, LocalizationFinishObservation]):
    """Tool for submitting final code localization results."""

    @classmethod
    def create(
        cls,
        conv_state,
        workspace_dir: str | None = None,
        **params
    ) -> Sequence["LocalizationFinishTool"]:
        """Create LocalizationFinishTool instance.

        Args:
            conv_state: Conversation state (provides workspace info)
            workspace_dir: Optional workspace directory override
            **params: Additional parameters

        Returns:
            A sequence containing a single LocalizationFinishTool instance.
        """
        # Get workspace from conv_state if not provided
        if workspace_dir is None and hasattr(conv_state, 'workspace'):
            workspace_dir = str(conv_state.workspace.working_dir)

        executor = LocalizationFinishExecutor(workspace_dir=workspace_dir)

        return [
            cls(
                action_type=LocalizationFinishAction,
                observation_type=LocalizationFinishObservation,
                description=TOOL_DESCRIPTION,
                executor=executor,
            )
        ]


@tool(name="localization_finish")
def _make_localization_finish_tool(conv_state) -> list[ToolDefinition]:
    """Create localization finish tool.

    This is a localization-specific finish tool that accepts structured locations
    and validates the output format.
    """
    return LocalizationFinishTool.create(conv_state)
