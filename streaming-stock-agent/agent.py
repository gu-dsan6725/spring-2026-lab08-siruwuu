"""Stock query agent using Strands and Yahoo Finance."""

import logging
import json
from typing import Dict, Any, List
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(
    filename: str
) -> str:
    """Load prompt from prompts directory.

    Args:
        filename: Name of the prompt file

    Returns:
        Prompt text content
    """
    prompt_path = PROMPTS_DIR / filename

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text()


def _get_stock_price(
    ticker: str
) -> Dict[str, Any]:
    """Get current stock price for a ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')

    Returns:
        Dictionary with stock price information
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        previous_close = info.get('previousClose')

        if current_price is None:
            return {
                "error": f"Could not retrieve price for {ticker}",
                "ticker": ticker.upper()
            }

        change = None
        change_percent = None
        if previous_close:
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100

        return {
            "ticker": ticker.upper(),
            "name": info.get('longName', ticker.upper()),
            "current_price": round(current_price, 2),
            "previous_close": round(previous_close, 2) if previous_close else None,
            # "change": round(change, 2) if change else None,
            # "change_percent": round(change_percent, 2) if change_percent else None,
            "change": round(change, 2) if change is not None else None,
            "change_percent": round(change_percent, 2) if change_percent is not None else None,
            "currency": info.get('currency', 'USD'),
            "market_state": info.get('marketState', 'UNKNOWN'),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {e}")
        return {
            "error": str(e),
            "ticker": ticker.upper()
        }


def _get_stock_history(
    ticker: str,
    days: int = 30
) -> Dict[str, Any]:
    """Get historical stock prices.

    Args:
        ticker: Stock ticker symbol
        days: Number of days of history to retrieve (default: 30)

    Returns:
        Dictionary with historical price data
    """
    try:
        stock = yf.Ticker(ticker.upper())
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return {
                "error": f"No historical data available for {ticker}",
                "ticker": ticker.upper()
            }

        # Get latest and earliest prices for period summary
        latest = hist.iloc[-1]
        earliest = hist.iloc[0]

        period_change = latest['Close'] - earliest['Close']
        period_change_percent = (period_change / earliest['Close']) * 100

        return {
            "ticker": ticker.upper(),
            "period_days": days,
            "start_date": hist.index[0].strftime('%Y-%m-%d'),
            "end_date": hist.index[-1].strftime('%Y-%m-%d'),
            "start_price": round(earliest['Close'], 2),
            "end_price": round(latest['Close'], 2),
            "period_change": round(period_change, 2),
            "period_change_percent": round(period_change_percent, 2),
            "high": round(hist['High'].max(), 2),
            "low": round(hist['Low'].min(), 2),
            "average_volume": int(hist['Volume'].mean()),
            "data_points": len(hist)
        }

    except Exception as e:
        logger.error(f"Error fetching history for {ticker}: {e}")
        return {
            "error": str(e),
            "ticker": ticker.upper()
        }


def _get_company_info(
    ticker: str
) -> Dict[str, Any]:
    """Get company information for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with company details
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        return {
            "ticker": ticker.upper(),
            "name": info.get('longName'),
            "sector": info.get('sector'),
            "industry": info.get('industry'),
            "description": info.get('longBusinessSummary'),
            "website": info.get('website'),
            "market_cap": info.get('marketCap'),
            "employees": info.get('fullTimeEmployees'),
            "country": info.get('country'),
            "exchange": info.get('exchange')
        }

    except Exception as e:
        logger.error(f"Error fetching company info for {ticker}: {e}")
        return {
            "error": str(e),
            "ticker": ticker.upper()
        }

def _compare_stocks(
    ticker1: str,
    ticker2: str
) -> Dict[str, Any]:
    """Compare two stocks using current price and basic company metrics.

    Args:
        ticker1: First stock ticker symbol
        ticker2: Second stock ticker symbol

    Returns:
        Dictionary comparing the two stocks
    """
    try:
        stock1 = yf.Ticker(ticker1.upper())
        stock2 = yf.Ticker(ticker2.upper())

        info1 = stock1.info
        info2 = stock2.info

        price1 = info1.get("currentPrice") or info1.get("regularMarketPrice")
        price2 = info2.get("currentPrice") or info2.get("regularMarketPrice")

        prev_close1 = info1.get("previousClose")
        prev_close2 = info2.get("previousClose")

        change1 = None
        change2 = None
        change_pct1 = None
        change_pct2 = None

        if price1 is not None and prev_close1:
            change1 = price1 - prev_close1
            change_pct1 = (change1 / prev_close1) * 100

        if price2 is not None and prev_close2:
            change2 = price2 - prev_close2
            change_pct2 = (change2 / prev_close2) * 100

        return {
            "ticker1": {
                "ticker": ticker1.upper(),
                "name": info1.get("longName", ticker1.upper()),
                "current_price": round(price1, 2) if price1 is not None else None,
                "previous_close": round(prev_close1, 2) if prev_close1 else None,
                "change": round(change1, 2) if change1 is not None else None,
                "change_percent": round(change_pct1, 2) if change_pct1 is not None else None,
                "market_cap": info1.get("marketCap"),
                "pe_ratio": info1.get("trailingPE"),
                "sector": info1.get("sector"),
                "currency": info1.get("currency", "USD"),
            },
            "ticker2": {
                "ticker": ticker2.upper(),
                "name": info2.get("longName", ticker2.upper()),
                "current_price": round(price2, 2) if price2 is not None else None,
                "previous_close": round(prev_close2, 2) if prev_close2 else None,
                "change": round(change2, 2) if change2 is not None else None,
                "change_percent": round(change_pct2, 2) if change_pct2 is not None else None,
                "market_cap": info2.get("marketCap"),
                "pe_ratio": info2.get("trailingPE"),
                "sector": info2.get("sector"),
                "currency": info2.get("currency", "USD"),
            },
            "summary": f"Comparison between {ticker1.upper()} and {ticker2.upper()}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error comparing stocks {ticker1} and {ticker2}: {e}")
        return {
            "error": str(e),
            "ticker1": ticker1.upper(),
            "ticker2": ticker2.upper()
        }

# Tool definitions for Strands agent
STOCK_TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price and basic information for a ticker symbol. Use this when the user asks about current price, real-time price, or latest price.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
                }
            },
            "required": ["ticker"]
        },
        "function": _get_stock_price
    },
    {
        "name": "get_stock_history",
        "description": "Get historical stock price data over a period. Use this when the user asks about price changes, trends, or historical performance.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days of history to retrieve (default: 30)",
                    "default": 30
                }
            },
            "required": ["ticker"]
        },
        "function": _get_stock_history
    },
    {
        "name": "get_company_info",
        "description": "Get detailed company information including sector, industry, and description. Use this when the user asks about the company itself, not just the stock price.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
                }
            },
            "required": ["ticker"]
        },
        "function": _get_company_info
    },
    {
        "name": "compare_stocks",
        "description": "Compare two stocks using current price, market cap, PE ratio, and sector information. Use this when the user asks to compare two companies or stocks.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker1": {
                    "type": "string",
                    "description": "First stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
                },
                "ticker2": {
                    "type": "string",
                    "description": "Second stock ticker symbol (e.g., NVDA, AMZN, GOOGL)"
                }
            },
            "required": ["ticker1", "ticker2"]
        },
        "function": _compare_stocks
    }
]


def get_system_prompt() -> str:
    """Get the system prompt for the stock agent."""
    return _load_prompt("system_prompt.txt")


def get_tool_by_name(
    tool_name: str
) -> Any:
    """Get tool function by name."""
    for tool in STOCK_TOOLS:
        if tool["name"] == tool_name:
            return tool["function"]
    return None


def execute_tool_call(
    tool_name: str,
    parameters: Dict[str, Any]
) -> str:
    """Execute a tool call and return JSON result."""
    tool_func = get_tool_by_name(tool_name)

    if not tool_func:
        return json.dumps({"error": f"Tool {tool_name} not found"})

    try:
        result = tool_func(**parameters)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return json.dumps({"error": str(e)})
