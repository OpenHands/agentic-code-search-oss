"""
Example: Using semantic search with a local vLLM-served model.

This demonstrates how to integrate semantic code search with a locally
served LLM through vLLM, avoiding external API dependencies.
"""

import asyncio
import os
from openhands.sdk import LLM, Agent, Conversation, AgentContext
from openhands.sdk.context.skills import Skill
from pydantic import SecretStr

async def main():
    # 1. Load semantic search skill
    skill = Skill.load(".openhands/skills/semantic-search.md")

    # 2. Configure LLM to use local vLLM server
    # vLLM should be running at http://localhost:8000
    # Start it with: vllm serve your-model-name --host 0.0.0.0 --port 8000
    llm = LLM(
        model="Qwen/Qwen3-8B",  
        base_url="http://localhost:8000/v1",
        api_key=SecretStr("EMPTY"),
        custom_llm_provider="openai",  
        temperature=0.0,
        max_output_tokens=16384,
        timeout=120,
        usage_id="vllm-agent"
    )

    # 3. Configure MCP server for semantic search
    mcp_config = {
        "mcpServers": {
            "semantic-code-search": {
                "command": "uv",
                "args": ["run", "python", "src/mcp_server/training_semantic_search_server.py"],
                "env": {}
            }
        }
    }

    # 4. Create agent context with skills
    context = AgentContext(skills=[skill])

    # 5. Create agent with local LLM and MCP tools
    agent = Agent(llm=llm, agent_context=context, mcp_config=mcp_config)

    # 6. Run conversation
    conversation = Conversation(agent=agent, workspace=".")

    print("Starting conversation with local vLLM model:")

    conversation.send_message(
        "Find code related to reward calculation and metrics in this repository"
    )
    await conversation.run()

    print("Agent Response:")
    print(conversation.agent_final_response())

    # 7. Show metrics
    print("LLM Metrics:")
    print(f"Total input tokens: {llm.metrics.accumulated_input_tokens}")
    print(f"Total output tokens: {llm.metrics.accumulated_output_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
