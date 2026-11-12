from typing import Literal

import ollama
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()
OpenBinaryMode = Literal["rb", "wb", "ab", "xb"]

MAX_ITERATIONS = 10
MODEL = "qwen2.5:3b"


# Tools
@traceable(name="tool")
def get_product_price(product: str) -> float:
    """Look yp the price of the a product in the catalog"""
    print(f"       >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product.lower(), 0)


@traceable(name="tool")
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


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "description": "Product name"}
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number"},
                    "discount_tier": {
                        "type": "string",
                        "enum": ["bronze", "silver", "gold"],
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# Agent
@traceable(name="Agent Loop")
def run_agent(question: str):
    tools = {"get_product_price": get_product_price, "apply_discount": apply_discount}

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
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
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n---- iteration {iteration} ------ ")
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\n Final Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool cal
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f" Tool '{tool_name}' not found")
        observation = tool_to_use(**tool_args)
        print(f" [Tool Result] {observation}")
        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})
    print("Error")


if __name__ == "__main__":
    print("Hello Langchain agent (.bind_tools)")
    result = run_agent("What is the price of a laptop after applying gold discount")
