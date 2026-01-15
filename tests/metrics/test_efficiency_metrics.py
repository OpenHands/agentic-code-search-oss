"""Tests for efficiency metrics computation."""

import pytest
from src.metrics.efficiency_metrics import (
    compute_token_metrics,
    compute_step_count,
    compute_tool_call_metrics,
    compute_all_efficiency_metrics,
)


# Mock data structures
MOCK_TOKEN_MESSAGE_1 = {
    "kind": "TokenEvent",
    "prompt_token_ids": [1, 2, 3, 4],  # 4 tokens
    "response_token_ids": [5, 6, 7],  # 3 tokens
}

MOCK_TOKEN_MESSAGE_2 = {
    "kind": "TokenEvent",
    "prompt_token_ids": [1, 2, 3, 4, 5, 6],  # 6 tokens
    "response_token_ids": [7, 8],  # 2 tokens
}

# ActionEvent mock data (the actual tool invocations)
MOCK_ACTION_EVENT_BASH = {
    "kind": "ActionEvent",
    "tool_name": "bash",
    "llm_response_id": "response_1",
}

MOCK_ACTION_EVENT_RESULT = {
    "kind": "ActionEvent",
    "tool_name": "result",
    "llm_response_id": "response_1",
}

MOCK_ACTION_EVENT_SEARCH = {
    "kind": "ActionEvent",
    "tool_name": "search",
    "llm_response_id": "response_2",
}

# Legacy mock data (kept for backward compatibility with old tests)
MOCK_ASSISTANT_MESSAGE_WITH_TOOLS = {
    "role": "assistant",
    "content": "I'll search for the file",
    "tool_calls": [
        {
            "function": {
                "name": "bash",
                "arguments": '{"command": "rg pattern"}'
            }
        },
        {
            "function": {
                "name": "result",
                "arguments": '{"file_paths": ["file.py"]}'
            }
        }
    ]
}

MOCK_ASSISTANT_MESSAGE_NO_TOOLS = {
    "role": "assistant",
    "content": "Here's the result",
}

MOCK_TOOL_RESPONSE = {
    "role": "tool",
    "content": "Search results here",
}


class TestComputeTokenMetrics:
    """Tests for compute_token_metrics function."""

    def test_empty_messages(self):
        """Test with empty messages list."""
        result = compute_token_metrics([])
        assert result["total_tokens"] == 0
        assert result["total_prompt_tokens"] == 0
        assert result["total_response_tokens"] == 0
        assert result["avg_prompt_tokens_per_step"] == 0.0
        assert result["avg_response_tokens_per_step"] == 0.0

    def test_no_token_events(self):
        """Test with messages but no TokenEvents."""
        messages = [MOCK_ASSISTANT_MESSAGE_NO_TOOLS, MOCK_TOOL_RESPONSE]
        result = compute_token_metrics(messages)
        assert result["total_tokens"] == 0

    def test_single_token_event(self):
        """Test with single TokenEvent."""
        messages = [MOCK_TOKEN_MESSAGE_1]
        result = compute_token_metrics(messages)
        assert result["total_tokens"] == 7  # 4 prompt + 3 response
        assert result["total_prompt_tokens"] == 4
        assert result["total_response_tokens"] == 3
        assert result["avg_prompt_tokens_per_step"] == 4.0
        assert result["avg_response_tokens_per_step"] == 3.0

    def test_multiple_token_events(self):
        """Test with multiple TokenEvents."""
        messages = [MOCK_TOKEN_MESSAGE_1, MOCK_TOKEN_MESSAGE_2]
        result = compute_token_metrics(messages)
        assert result["total_tokens"] == 15  # (4+3) + (6+2)
        assert result["total_prompt_tokens"] == 10  # 4 + 6
        assert result["total_response_tokens"] == 5  # 3 + 2
        assert result["avg_prompt_tokens_per_step"] == 5.0  # 10/2
        assert result["avg_response_tokens_per_step"] == 2.5  # 5/2

    def test_mixed_messages(self):
        """Test with mix of TokenEvents and other messages."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,
            MOCK_ASSISTANT_MESSAGE_NO_TOOLS,
            MOCK_TOKEN_MESSAGE_2,
            MOCK_TOOL_RESPONSE,
        ]
        result = compute_token_metrics(messages)
        # Should only count the two TokenEvents
        assert result["total_tokens"] == 15
        assert result["total_prompt_tokens"] == 10
        assert result["total_response_tokens"] == 5


class TestComputeStepCount:
    """Tests for compute_step_count function."""

    def test_empty_messages(self):
        """Test with empty messages list."""
        assert compute_step_count([]) == 0

    def test_no_token_events(self):
        """Test with messages but no TokenEvents."""
        messages = [MOCK_ASSISTANT_MESSAGE_NO_TOOLS, MOCK_TOOL_RESPONSE]
        assert compute_step_count(messages) == 0

    def test_single_token_event(self):
        """Test with single TokenEvent."""
        messages = [MOCK_TOKEN_MESSAGE_1]
        assert compute_step_count(messages) == 1

    def test_multiple_token_events(self):
        """Test with multiple TokenEvents."""
        messages = [MOCK_TOKEN_MESSAGE_1, MOCK_TOKEN_MESSAGE_2]
        assert compute_step_count(messages) == 2

    def test_mixed_messages(self):
        """Test with mix of TokenEvents and other messages."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,
            MOCK_ASSISTANT_MESSAGE_NO_TOOLS,
            MOCK_TOKEN_MESSAGE_2,
            MOCK_TOOL_RESPONSE,
        ]
        assert compute_step_count(messages) == 2


class TestComputeToolCallMetrics:
    """Tests for compute_tool_call_metrics function."""

    def test_empty_messages(self):
        """Test with empty messages list."""
        result = compute_tool_call_metrics([])
        assert result["total_tool_calls"] == 0
        assert result["avg_tool_calls_per_step"] == 0.0
        assert result["tool_call_breakdown"] == {}

    def test_no_tool_calls(self):
        """Test with messages but no tool calls."""
        messages = [MOCK_TOKEN_MESSAGE_1]
        result = compute_tool_call_metrics(messages)
        assert result["total_tool_calls"] == 0
        assert result["avg_tool_calls_per_step"] == 0.0

    def test_single_action_event(self):
        """Test with single action event (tool call)."""
        messages = [MOCK_TOKEN_MESSAGE_1, MOCK_ACTION_EVENT_BASH]
        result = compute_tool_call_metrics(messages)
        assert result["total_tool_calls"] == 1
        assert result["avg_tool_calls_per_step"] == 1.0  # 1 tool / 1 step
        assert result["tool_call_breakdown"]["bash"] == 1

    def test_multiple_action_events_same_turn(self):
        """Test with multiple action events in the same turn."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,
            MOCK_ACTION_EVENT_BASH,
            MOCK_ACTION_EVENT_RESULT,
        ]
        result = compute_tool_call_metrics(messages)
        assert result["total_tool_calls"] == 2
        assert result["avg_tool_calls_per_step"] == 2.0  # 2 tools / 1 step
        assert result["tool_call_breakdown"]["bash"] == 1
        assert result["tool_call_breakdown"]["result"] == 1

    def test_multiple_turns_with_tools(self):
        """Test with multiple turns, each with tool calls."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,
            MOCK_ACTION_EVENT_BASH,
            MOCK_ACTION_EVENT_RESULT,
            MOCK_TOKEN_MESSAGE_2,
            MOCK_ACTION_EVENT_SEARCH,
        ]
        result = compute_tool_call_metrics(messages)
        assert result["total_tool_calls"] == 3  # 2 + 1 tools
        assert result["avg_tool_calls_per_step"] == 1.5  # 3 tools / 2 steps
        assert result["tool_call_breakdown"]["bash"] == 1
        assert result["tool_call_breakdown"]["result"] == 1
        assert result["tool_call_breakdown"]["search"] == 1

    def test_mixed_turns_with_and_without_tools(self):
        """Test with mix of turns with and without tools."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,
            MOCK_ACTION_EVENT_BASH,
            MOCK_TOKEN_MESSAGE_2,  # No tools in this turn
        ]
        result = compute_tool_call_metrics(messages)
        assert result["total_tool_calls"] == 1
        assert result["avg_tool_calls_per_step"] == 0.5  # 1 tool / 2 steps


class TestComputeAllEfficiencyMetrics:
    """Tests for compute_all_efficiency_metrics function."""

    def test_empty_messages(self):
        """Test with empty messages list."""
        result = compute_all_efficiency_metrics(
            messages=[],
            wall_clock_duration=10.5,
        )
        assert result["tokens"] == 0
        assert result["steps"] == 0
        assert result["avg_tool_calls_per_step"] == 0.0
        assert result["wall_clock_duration"] == 10.5

    def test_complete_trajectory(self):
        """Test with complete trajectory including tokens, steps, and tools."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,  # Step 1: 7 tokens
            MOCK_ACTION_EVENT_BASH,  # tool call 1
            MOCK_ACTION_EVENT_RESULT,  # tool call 2
            MOCK_TOKEN_MESSAGE_2,  # Step 2: 8 tokens
            MOCK_ACTION_EVENT_BASH,  # tool call 3
            MOCK_ACTION_EVENT_RESULT,  # tool call 4
        ]
        result = compute_all_efficiency_metrics(
            messages=messages,
            wall_clock_duration=15.5,
            start_timestamp="2025-01-01T10:00:00",
            end_timestamp="2025-01-01T10:00:15",
        )

        # Check core metrics
        assert result["tokens"] == 15  # 7 + 8
        assert result["steps"] == 2
        assert result["avg_tool_calls_per_step"] == 2.0  # 4 tools / 2 steps
        assert result["wall_clock_duration"] == 15.5

        # Check extended metrics
        assert result["total_prompt_tokens"] == 10
        assert result["total_response_tokens"] == 5

    def test_without_timestamps(self):
        """Test that timestamps are optional."""
        messages = [MOCK_TOKEN_MESSAGE_1]
        result = compute_all_efficiency_metrics(
            messages=messages,
            wall_clock_duration=5.0,
        )
        # Timestamps are not included in the current implementation
        assert "start_timestamp" not in result
        assert "end_timestamp" not in result

    def test_minimal_trajectory(self):
        """Test with minimal valid trajectory (single step, no tools)."""
        messages = [MOCK_TOKEN_MESSAGE_1]
        result = compute_all_efficiency_metrics(
            messages=messages,
            wall_clock_duration=2.0,
        )
        assert result["tokens"] == 7
        assert result["steps"] == 1
        assert result["avg_tool_calls_per_step"] == 0.0
        assert result["wall_clock_duration"] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
