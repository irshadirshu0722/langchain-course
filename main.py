import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

# @tool
# def search(query:str) -> str:
#     """
#     Tool that searches over internet

#     Args:
#         Query: The query to search for
#     Returns:
#         the search result
#     """
#     print("search for",query)
#     return tavily.search(query=query)

search = TavilySearch()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.7,
)

tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
