from dotenv import load_dotenv
from langchain_community.tools import tool
import requests
import os
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
# from langchain.agents.react import create_react_agent
# from langchain.agents.agent_iterator import AgentExecutor
from langchain.agents import create_agent
from langchain_classic import hub


load_dotenv()

# get the api key from environment variable.
API_KEY = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)
access_key = os.getenv("access_key")

search_tool = DuckDuckGoSearchRun()

# Simple system prompt
system_prompt = (
    "You are an intelligent agent that can use tools to answer user queries. "
    "Use the provided tools when necessary and return a helpful response."
)

@tool
def get_weather_data(city:str)->str:
    """
    This function fetches the current weather data for a given city. 
    """
    url = f"https://api.weatherstack.com/current?access_key={access_key}&query={city}"
    response = requests.get(url)
    return response.json()





graph = create_agent(model=model,tools=[search_tool, get_weather_data],system_prompt=system_prompt,name="Population discovering agent",)

inputs = {"messages": [{"role": "user", "content": "what is the capital of France? then find the current weather condition"}]}

for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)