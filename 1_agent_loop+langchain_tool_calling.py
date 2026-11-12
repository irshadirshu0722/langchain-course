from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langsmith import traceable

load_dotenv()
OpenBinaryMode = Literal["rb", "wb", "ab", "xb"]

MAX_ITERATIONS = 10
MODEL = "gemini-3.5-flash-lite"


# Tools
@tool
def get_product_price(product: str) -> float:
    """Look yp the price of the a product in the catalog"""
    print(f"       >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product.lower(), 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """
    print(
        f"       >>Executing apply_discount(price={price},discount_tier='{discount_tier}')"
    )
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentage.get(discount_tier, 0)
    return round(price * (1 - (discount / 100)), 2)


# Agent
@traceable(name="Langchain Agent loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"google_genai:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "You have access to product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES - you much follow these exactly: \n"
                "1: NEVER guess or assume any product price"
                "2: You must call get_product_price. Pass the exact price"
                "3: Only call apply_discount after you have received a price from get_product_price. Pass the exact price"
                "Do not pass a made-up number.\n"
                "Always use he apply_discount to calculate discount, Never calculate yourself"
                "4.If the user does not specify a discount tier. Ask them which tier to use - do NOT assume one"
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n---- iteration {iteration} ------ ")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\n Final Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool cal
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f" Tool '{tool_name}' not found")
        observation = tool_to_use.invoke(tool_args)
        print(f" [Tool Result] {observation}")
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print("Error")


if __name__ == "__main__":
    print("Hello Langchain agent (.bind_tools)")
    result = run_agent("What is the price of a laptop after applying gold discount")
