"""Financial Optimization Orchestrator Agent.

This agent demonstrates the orchestrator-workers pattern using Claude Agent SDK.
It fetches financial data from MCP servers and coordinates specialized sub-agents
to provide comprehensive financial optimization recommendations.
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    PermissionResultAllow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


DATA_DIR: Path = Path(__file__).parent.parent / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw_data"
AGENT_OUTPUTS_DIR: Path = DATA_DIR / "agent_outputs"


def _ensure_directories():
    """Ensure all required directories exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(
    data: dict,
    filename: str
):
    """Save data to JSON file."""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved data to {filepath}")


def _load_prompt(filename: str) -> str:
    """Load prompt from prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / filename
    return prompt_path.read_text(encoding="utf-8")


async def _auto_approve_all(
    tool_name: str,
    input_data: dict,
    context
):
    """Auto-approve all tools without prompting."""
    logger.debug(f"Auto-approving tool: {tool_name}")
    return PermissionResultAllow()


def _detect_subscriptions(
    bank_transactions: list[dict],
    credit_card_transactions: list[dict]
) -> list[dict]:
    """Detect subscription services from recurring transactions.

    Logic:
    1. Filter transactions marked as recurring
    2. Keep outflow transactions only
    3. Extract service name, amount, frequency
    4. Return normalized subscription list

    Args:
        bank_transactions: List of bank transaction dicts
        credit_card_transactions: List of credit card transaction dicts

    Returns:
        List of subscription dictionaries with service name, amount, frequency
    """
    subscriptions: list[dict] = []
    all_transactions = bank_transactions + credit_card_transactions

    for tx in all_transactions:
        if tx.get("recurring") is not True:
            continue

        amount = tx.get("amount", 0)
        if not isinstance(amount, (int, float)):
            continue

        # Keep only money going out
        if amount >= 0:
            continue

        service = (
            tx.get("merchant")
            or tx.get("description")
            or tx.get("name")
            or "Unknown Service"
        )

        frequency = tx.get("frequency", "monthly")

        subscriptions.append(
            {
                "service": service,
                "amount": abs(float(amount)),
                "frequency": frequency,
            }
        )

    return subscriptions


async def _fetch_financial_data(
    username: str,
    start_date: str,
    end_date: str
) -> tuple[dict, dict]:
    """Fetch data from Bank and Credit Card MCP servers."""
    logger.info(f"Fetching financial data for {username} from {start_date} to {end_date}")

    mcp_servers = {
        "Bank Account Server": {
            "type": "http",
            "url": "http://127.0.0.1:5001/mcp"
        },
        "Credit Card Server": {
            "type": "http",
            "url": "http://127.0.0.1:5002/mcp"
        }
    }

    
    
    options = ClaudeAgentOptions(
        mcp_servers=mcp_servers,
        allowed_tools=[
            "mcp__Bank_Account_Server__get_bank_transactions",
            "mcp__Credit_Card_Server__get_credit_card_transactions",
        ],
        can_use_tool=_auto_approve_all,
        cwd=Path(__file__).parent.parent,
    )

    fetch_prompt = f"""
Use the MCP tools to fetch financial data for username "{username}"
from {start_date} to {end_date}.

You must do exactly these two tool calls:
1. get_bank_transactions
2. get_credit_card_transactions

Do not write any files.
Do not summarize.
Return only valid JSON in exactly this format:

{{
  "bank_data": <result of get_bank_transactions>,
  "credit_card_data": <result of get_credit_card_transactions>
}}
"""

    collected_text_parts: list[str] = []

    async with ClaudeSDKClient(options=options) as client:
        await client.query(fetch_prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
                        collected_text_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                print()
                logger.info(f"Fetch duration: {message.duration_ms}ms")
                logger.info(f"Fetch cost: ${message.total_cost_usd:.4f}")
                break

    raw_text = "".join(collected_text_parts).strip()

    # 去掉 ```json ... ``` code fence
    if raw_text.startswith("```json"):
        raw_text = raw_text[len("```json"):].strip()
    if raw_text.startswith("```"):
        raw_text = raw_text[len("```"):].strip()
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3].strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            parsed = json.loads(raw_text[start_idx:end_idx + 1])
        else:
            raise ValueError("Could not parse MCP fetch response as JSON")

    bank_data = parsed.get("bank_data", {})
    credit_card_data = parsed.get("credit_card_data", {})

    _save_json(bank_data, "bank_transactions.json")
    _save_json(credit_card_data, "credit_card_transactions.json")

    return bank_data, credit_card_data


async def _run_orchestrator(
    username: str,
    start_date: str,
    end_date: str,
    user_query: str
):
    """Main orchestrator agent logic.

    Args:
        username: Username for the account
        start_date: Start date for analysis
        end_date: End date for analysis
        user_query: User's financial question/request
    """
    logger.info("Starting financial optimization orchestrator")
    logger.info(f"User query: {user_query}")

    _ensure_directories()

    # Step 1: Fetch financial data from MCP servers
    bank_data, credit_card_data = await _fetch_financial_data(
        username,
        start_date,
        end_date
    )

    # Step 2: Initial analysis
    logger.info("Performing initial analysis...")

    bank_transactions = bank_data.get("transactions", [])
    credit_card_transactions = credit_card_data.get("transactions", [])

    subscriptions = _detect_subscriptions(
        bank_transactions,
        credit_card_transactions
    )

    logger.info(f"Detected {len(subscriptions)} subscriptions")

    # Same MCP config reused by orchestrator
    mcp_servers = {
        "Bank Account Server": {
            "type": "http",
            "url": "http://127.0.0.1:5001/mcp",
        },
        "Credit Card Server": {
            "type": "http",
            "url": "http://127.0.0.1:5002/mcp",
        },
    }

    # Step 3: Define sub-agents
    research_agent = AgentDefinition(
        description="Research cheaper alternatives for subscriptions and services",
        prompt=_load_prompt("research_agent_prompt.txt"),
        tools=["write"],
        model="haiku",
    )

    negotiation_agent = AgentDefinition(
        description="Create negotiation strategies and scripts for bills and services",
        prompt=_load_prompt("negotiation_agent_prompt.txt"),
        tools=["write"],
        model="haiku",
    )

    tax_agent = AgentDefinition(
        description="Identify tax-deductible expenses and optimization opportunities",
        prompt=_load_prompt("tax_agent_prompt.txt"),
        tools=["write"],
        model="haiku",
    )

    agents = {
        "research_agent": research_agent,
        "negotiation_agent": negotiation_agent,
        "tax_agent": tax_agent,
    }

    # Step 4: Configure orchestrator agent with sub-agents
    working_dir = Path(__file__).parent.parent

    options = ClaudeAgentOptions(
        model="sonnet",
        system_prompt=_load_prompt("orchestrator_system_prompt.txt"),
        mcp_servers=mcp_servers,
        agents=agents,
        can_use_tool=_auto_approve_all,
        cwd=working_dir,
    )

    # Step 5: Run orchestrator with Claude Agent SDK
    subscriptions_json = json.dumps(subscriptions, indent=2)

    prompt = f"""Analyze my financial data and answer this user request: {user_query}

    The financial data has ALREADY been fetched for you.
    Do NOT fetch bank or credit card transactions again unless absolutely necessary.

    Available local files:
    - data/raw_data/bank_transactions.json
    - data/raw_data/credit_card_transactions.json

    Summary:
    - {len(bank_transactions)} bank transactions
    - {len(credit_card_transactions)} credit card transactions
    - {len(subscriptions)} identified subscriptions

    Detected subscriptions:
    {subscriptions_json}

    Your tasks:
    1. Read the existing raw data files if needed
    2. Identify opportunities for savings
    3. Delegate cheaper alternative research to research_agent
    4. Delegate bill/service negotiation strategy creation to negotiation_agent
    5. Delegate tax optimization analysis to tax_agent
    6. Ensure the sub-agents write their outputs to:
    - data/agent_outputs/research_results.md
    - data/agent_outputs/negotiation_scripts.md
    - data/agent_outputs/tax_analysis.md
    7. Read those outputs after they are created
    8. Write the final synthesized report to:
    - data/final_report.md

    Important constraints:
    - Prefer using the already fetched local JSON files
    - Do not re-fetch transactions from MCP unless required
    - Run sub-agents in parallel when possible
    - Provide concrete and actionable savings recommendations
    """

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
            elif isinstance(message, ResultMessage):
                print()
                logger.info(f"Duration: {message.duration_ms}ms")
                logger.info(f"Cost: ${message.total_cost_usd:.4f}")
                break

    logger.info("Orchestration complete. Check data/final_report.md for results.")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Financial Optimization Orchestrator Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Basic analysis
    uv run python financial_orchestrator.py \\
        --username john_doe \\
        --start-date 2026-01-01 \\
        --end-date 2026-01-31 \\
        --query "How can I save $500 per month?"

    # Subscription analysis
    uv run python financial_orchestrator.py \\
        --username jane_smith \\
        --start-date 2026-01-01 \\
        --end-date 2026-01-31 \\
        --query "Analyze my subscriptions and find better deals"
"""
    )

    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Username for account (john_doe or jane_smith)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User's financial question or request"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = _parse_args()

    # Optional date validation
    datetime.strptime(args.start_date, "%Y-%m-%d")
    datetime.strptime(args.end_date, "%Y-%m-%d")

    await _run_orchestrator(
        username=args.username,
        start_date=args.start_date,
        end_date=args.end_date,
        user_query=args.query
    )


if __name__ == "__main__":
    asyncio.run(main())