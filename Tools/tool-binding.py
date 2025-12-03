from dotenv import load_dotenv
import os
from langchain_core.tools import tool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

# get the api key from environment variable.
API_KEY = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY, temperature=1)


class InputFormat(BaseModel):
    num1:int = Field(...,description="First number to be added")
    num2:int = Field(...,description="Second number to be added")


def add_two_numbers(num1:int, num2:int) ->int:
    """
    add given two numbers and return the result
    """ 
    return num1 + num2

tool_with_schema = StructuredTool.from_function(name="add_two_numbers",
               description="add given two numbers and return the result",
               func=add_two_numbers, 
               args_schema=InputFormat)

# result = tool_with_schema.invoke({'num1':4,'num2':6})
# print(result)



llm_with_tool = model.bind_tools([add_two_numbers])
# print(f"LLM with tools: {llm_with_tool}")

# result = llm_with_tool.invoke("Hey!...")
result = llm_with_tool.invoke("Can you add 4 and 6")
print(result)
print(result.tool_calls[0]['args'])

response = tool_with_schema.invoke(result.tool_calls[0]['args'])
print(response)
response_from_tool_call = tool_with_schema.invoke({'name': 'add_two_numbers', 'args': {'num2': 6, 'num1': 4}, 'id': '42aee25b-810d-4c9d-9db4-dd48b523ac4c', 'type': 'tool_call'})

print(f"Response from tool call : {response_from_tool_call}")
# tool_response = tool.invoke({'name': 'add_two_numbers', 'args': {'num2': 6, 'num1': 4}, 'id': '075fb290-603a-4582-bf37-f2620d1487a2', 'type': 'tool_call'}) 
# print(f"Tool Response: {tool_response}")


# LLM tool call doesn't mean an LLM will call or execute a tool on behalf of you!...
# It's the programmer that handles the tool execution.