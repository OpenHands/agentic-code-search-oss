# SkyRL Experiment Configuration Guide

This directory contains experiment configuration files for training agents with SkyRL. Each YAML file defines an experiment setup with specific tools, rewards, and prompts.

## Usage

```bash
DATA_PATH=<Absolute Path to Data>

bash scripts/run_async_training.sh \
    -m Qwen/Qwen3-4B \
    -o "+generator.exp_config=configs/skyrl-experiments/read-only.yaml" \
    -d $DATA_PATH \
    2>&1 | tee training.log
```

## Configuration File Structure

Each experiment config file follows this structure:

```yaml
name: "experiment_name"
description: "Brief description of the experiment"

reward:
  - fn: reward_function_1
  - fn: reward_function_2

tools:
  - tool_name_1
  - tool_name_2

prompts:
  system_prompt: "templates/system_prompt.j2"
  user_prompt: "templates/user_prompt.j2"
```

### Fields

#### `name` (optional)
- **Type**: String
- **Description**: A unique identifier for the experiment
- **Example**: `"read_only_tools"`

#### `description` (optional)
- **Type**: String
- **Description**: A human-readable description of what the experiment tests
- **Example**: `"The agent only has access to read only tools"`

#### `reward` (required)
- **Type**: List of reward function specifications
- **Description**: Defines the reward functions used to evaluate agent performance during training
- **Format**: Each item should have a `fn` key with the reward function name
- **Example**:
  ```yaml
  reward:
    - fn: tool_use_reward
    - fn: turn_efficiency
  ```

#### `tools` (required)
- **Type**: List of tool names
- **Description**: Specifies which tools the agent has access to during the experiment
- **Options**: Can be default OpenHands tools, custom tools, or toolsets
- **Example**:
  ```yaml
  tools:
    - terminal
    - grep
    - glob
  ```

#### `prompts` (required)
- **Type**: Object with `system_prompt` and `user_prompt` keys
- **Description**: Specifies the Jinja2 template files for system and user prompts
- **Location**: Templates should be placed in `src/prompts/templates/`
- **Format**: Paths are relative to `src/prompts/`
- **Example**:
  ```yaml
  prompts:
    system_prompt: "templates/system_prompt.j2"
    user_prompt: "templates/file_localization.j2"
  ```

## Default OpenHands Tools

The following tools are built into OpenHands and can be used directly in your config:

- `apply_patch` - Apply code patches to files
- `browser_use` - Interact with web browsers
- `delegate` - Delegate tasks to sub-agents
- `file_editor` - Edit files with various operations
- `glob` - Search for files by name patterns
- `grep` - Search file contents using regex
- `planning_file_editor` - File editor with planning capabilities
- `preset` - Use predefined tool presets
- `task_tracker` - Track and manage tasks
- `terminal` - Execute shell commands
- `tom_consult` - Consult theory of mind models

## Registering Custom Tools

To create and register a custom tool:

### 1. Create a Tool File

Create a new Python file in `src/tools/` (e.g., `src/tools/my_custom_tool.py`):

```python
from src.tools import tool
from pydantic import Field
from collections.abc import Sequence
from openhands.sdk import (
    Action,
    Observation,
    TextContent,
    ToolDefinition,
)
from openhands.sdk.tool import ToolExecutor

# Define your Action class
class MyCustomAction(Action):
    param1: str = Field(description="Description of parameter")
    param2: int = Field(default=10, description="Optional parameter")

# Define your Observation class
class MyCustomObservation(Observation):
    result: str = ""
    
    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [TextContent(text=self.result)]

# Define your Executor
class MyCustomExecutor(ToolExecutor[MyCustomAction, MyCustomObservation]):
    def __call__(self, action: MyCustomAction, conversation=None) -> MyCustomObservation:
        # Implement your tool logic here
        result = f"Processed {action.param1} with {action.param2}"
        return MyCustomObservation(result=result)

# Define your Tool
class MyCustomTool(ToolDefinition[MyCustomAction, MyCustomObservation]):
    @classmethod
    def create(cls, conv_state) -> Sequence[ToolDefinition]:
        executor = MyCustomExecutor()
        return [
            cls(
                description="Description of what your tool does",
                action_type=MyCustomAction,
                observation_type=MyCustomObservation,
                executor=executor,
            )
        ]

# Register the tool
@tool(name="my_custom_tool")
def _make_my_custom_tool(conv_state) -> list[ToolDefinition]:
    return MyCustomTool.create(conv_state)
```

### 2. Use the Tool in Your Config

Once registered, simply add the tool name to your experiment config:

```yaml
tools:
  - my_custom_tool
  - terminal
```

### Creating Toolsets

You can also create toolsets that bundle multiple tools together (see `bash_and_grep_toolset` in `src/tools/example_custom_tool.py`):

```python
@tool(name="my_toolset")
def _make_my_toolset(conv_state) -> list[ToolDefinition]:
    """Create multiple tools that share resources."""
    terminal_executor = TerminalExecutor(working_dir=conv_state.workspace.working_dir)
    
    tool1 = Tool1.create(conv_state, executor=terminal_executor)[0]
    tool2 = Tool2.create(conv_state, executor=terminal_executor)[0]
    
    return [tool1, tool2]
```

## System and User Prompts

Prompts are defined using Jinja2 templates and should be placed in `src/prompts/templates/`.

### Available Template Files

- `system_prompt.j2` - Default system prompt
- `file_localization.j2` - User prompt for file localization tasks
- `file_module.j2` - User prompt for file/module tasks
- `file_module_parallel_tools.j2` - User prompt with parallel tool usage
- `system_message_search.j2` - System prompt for search tasks
- `default.j2` - Default user prompt

### Creating Custom Prompts

1. Create a new Jinja2 template file in `src/prompts/templates/`:

```jinja2
{# templates/my_custom_prompt.j2 #}
You are an AI assistant specialized in {{ task_type }}.

Your goal is to: {{ goal }}

Available tools:
{% for tool in tools %}
- {{ tool }}
{% endfor %}

Please proceed with the task.
```

2. Reference it in your experiment config:

```yaml
prompts:
  system_prompt: "templates/system_prompt.j2"
  user_prompt: "templates/my_custom_prompt.j2"
```

### Template Variables

Templates have access to various context variables provided by the training system, including:
- `task_type` - The type of task being performed
- `goal` - The specific goal for the episode
- `tools` - List of available tools
- `workspace` - Workspace information
- And other context-specific variables

## Example Configurations

### Example 1: Read-Only Tools
```yaml
name: "read_only_tools"
description: "The agent only has access to read only tools"

reward:
  - fn: tool_use_reward
  - fn: turn_efficiency

tools:
  - glob
  - grep
  - terminal

prompts:
  system_prompt: "templates/system_prompt.j2"
  user_prompt: "templates/file_localization.j2"
```

### Example 2: Terminal Only
```yaml
name: "terminal_tool_only"
description: "The agent only has access to the terminal tool"

reward:
  - fn: tool_use_reward
  - fn: turn_efficiency

tools:
  - terminal

prompts:
  system_prompt: "templates/system_prompt.j2"
  user_prompt: "templates/file_localization.j2"
```

### Example 3: Custom Toolset
```yaml
name: "bash_and_grep"
description: "Agent uses bash and grep toolset with shared executor"

reward:
  - fn: tool_use_reward
  - fn: turn_efficiency

tools:
  - bash_and_grep_toolset

prompts:
  system_prompt: "templates/system_prompt.j2"
  user_prompt: "templates/file_localization.j2"
```
