from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

load_dotenv()
products = {
    "laptop": 200,
    "mobile": 500,
}
discount = {"silver": 15, "bronze": 5, "gold": 25}


@tool
def get_product(name: str) -> int | None:
    """
    Tool that find the product by name

    Args:
      name = Name of the product in string
    Returns:
      The price in number if exist, else None
    """
    return products.get(name.lower())


@tool
def get_discount_percent(tier: str) -> int | None:
    """
    Tool that find the tier discount by tier name

      Args:
        tier = Name of the tier in string
      Returns:
        The discount of tier in percent, if exist else None
    """
    return discount.get(tier)


tools = [get_discount_percent, get_product]


class DiscountResponse(BaseModel):
    discount_price: float


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.7,
)

agent = create_agent(model=llm, tools=tools, response_format=DiscountResponse)

response = agent.invoke(
    {
        "messages": HumanMessage(
            content="What is discount price of laptop with tier silver"
        )
    }
)
print(response)
