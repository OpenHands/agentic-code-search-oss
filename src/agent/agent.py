import json

from openhands.sdk.agent.utils import (
    make_llm_completion,
    prepare_llm_messages,
)
from openhands.sdk.conversation import (
    ConversationCallbackType,
    ConversationState,
    ConversationTokenCallbackType,
    LocalConversation,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import (
    ActionEvent,
    MessageEvent,
)
from openhands.sdk.event.condenser import Condensation, CondensationRequest
from openhands.sdk.llm import (
    Message,
    TextContent,
)
from openhands.sdk.llm.exceptions import (
    FunctionCallValidationError,
    LLMContextWindowExceedError,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.observability.laminar import (
    maybe_init_laminar,
    observe,
)

from openhands.sdk import Agent


logger = get_logger(__name__)
maybe_init_laminar()


class CustomAgent(Agent):

    @observe(name="agent.step", ignore_inputs=["state", "on_event"])
    def step(
        self,
        conversation: LocalConversation,
        on_event: ConversationCallbackType,
        on_token: ConversationTokenCallbackType | None = None,
    ) -> None:
        state = conversation.state
        # Check for pending actions (implicit confirmation)
        # and execute them before sampling new actions.
        pending_actions = ConversationState.get_unmatched_actions(state.events)
        if pending_actions:
            logger.info(
                "Confirmation mode: Executing %d pending action(s)",
                len(pending_actions),
            )
            self._execute_actions(conversation, pending_actions, on_event)
            return

        # Prepare LLM messages using the utility function
        _messages_or_condensation = prepare_llm_messages(
            state.events, condenser=self.condenser
        )

        # Process condensation event before agent sampels another action
        if isinstance(_messages_or_condensation, Condensation):
            on_event(_messages_or_condensation)
            return

        _messages = _messages_or_condensation

        logger.debug(
            "Sending messages to LLM: "
            f"{json.dumps([m.model_dump() for m in _messages[1:]], indent=2)}"
        )

        try:
            llm_response = make_llm_completion(
                self.llm,
                _messages,
                tools=list(self.tools_map.values()),
                on_token=on_token,
            )
        except FunctionCallValidationError as e:
            logger.warning(f"LLM generated malformed function call: {e}")
            error_message = MessageEvent(
                source="user",
                llm_message=Message(
                    role="user",
                    content=[TextContent(text=str(e))],
                ),
            )
            on_event(error_message)
            return
        except LLMContextWindowExceedError as e:
            # If condenser is available and handles requests, trigger condensation
            if (
                self.condenser is not None
                and self.condenser.handles_condensation_requests()
            ):
                logger.warning(
                    "LLM raised context window exceeded error, triggering condensation"
                )
                on_event(CondensationRequest())
                return
            # No condenser available or doesn't handle requests; log helpful warning
            self._log_context_window_exceeded_warning()
            raise e

        # LLMResponse already contains the converted message and metrics snapshot
        message: Message = llm_response.message

        # Manually extract reasoning content
        # if embedded within message
        content = message.content[0].text
        if "</think>" in content:
            reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
            content = content.split('</think>')[-1].lstrip('\n')
            message.content[0].text = content
            message.reasoning_content = reasoning_content

        # Check if this is a reasoning-only response (e.g., from reasoning models)
        # or a message-only response without tool calls
        has_reasoning = (
            message.responses_reasoning_item is not None
            or message.reasoning_content is not None
            or (message.thinking_blocks and len(message.thinking_blocks) > 0)
        )
        has_content = any(
            isinstance(c, TextContent) and c.text.strip() for c in message.content
        )

        if message.tool_calls and len(message.tool_calls) > 0:
            if not all(isinstance(c, TextContent) for c in message.content):
                logger.warning(
                    "LLM returned tool calls but message content is not all "
                    "TextContent - ignoring non-text content"
                )

            # Generate unique batch ID for this LLM response
            thought_content = [c for c in message.content if isinstance(c, TextContent)]

            action_events: list[ActionEvent] = []
            for i, tool_call in enumerate(message.tool_calls):
                action_event = self._get_action_event(
                    tool_call,
                    llm_response_id=llm_response.id,
                    on_event=on_event,
                    security_analyzer=state.security_analyzer,
                    thought=thought_content
                    if i == 0
                    else [],  # Only first gets thought
                    # Only first gets reasoning content
                    reasoning_content=message.reasoning_content if i == 0 else None,
                    # Only first gets thinking blocks
                    thinking_blocks=list(message.thinking_blocks) if i == 0 else [],
                    responses_reasoning_item=message.responses_reasoning_item
                    if i == 0
                    else None,
                )
                if action_event is None:
                    continue
                action_events.append(action_event)

            # Handle confirmation mode - exit early if actions need confirmation
            if self._requires_user_confirmation(state, action_events):
                return

            if action_events:
                self._execute_actions(conversation, action_events, on_event)

            # Emit VLLM token ids if enabled before returning
            self._maybe_emit_vllm_tokens(llm_response, on_event)
            return

        # No tool calls - emit message event for reasoning or content responses
        if not has_reasoning and not has_content:
            logger.warning("LLM produced empty response - continuing agent loop")

        msg_event = MessageEvent(
            source="agent",
            llm_message=message,
            llm_response_id=llm_response.id,
        )
        on_event(msg_event)

        # Emit VLLM token ids if enabled
        self._maybe_emit_vllm_tokens(llm_response, on_event)

        # Finish conversation if LLM produced content (awaits user input)
        # Continue if only reasoning without content (e.g., GPT-5 codex thinking)
        if has_content:
            logger.debug("LLM produced a message response - awaits user input")
            state.execution_status = ConversationExecutionStatus.FINISHED
            return
