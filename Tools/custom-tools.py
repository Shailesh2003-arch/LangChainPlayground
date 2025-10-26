# Custom tools in langchain...

# Creating a custom tool in langchain is a 3 step process...

# 1 step: For your tool to be executed - you need to define a function.
# 2 step: add type hints.
# 3 step: add tool decorator.

from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field


# @tool
# def add(a:int, b:int)->int:
#     """
#     Add two numbers and return its result
#     """
#     return a + b

# result = add.invoke({"a":3,"b":5})
# print(result)

# print(add.name)
# print(add.description)
# print(add.args)


# Now there exist different ways to create a tool...
# one of them was @tool decorator


# using Pydantic Class...

class MultiplyInput(BaseModel):
    num1:int = Field(...,description="First number to multiply")
    num2:int = Field(...,description="Second number to multiply")


def multiply_func(num1:int, num2:int)->int:
    return num1*num2



multiply_tool = StructuredTool.from_function(
    name="Multiply",
    func=multiply_func,
    args_schema=MultiplyInput,
    description="Multiply given numbers"

)


result = multiply_tool.invoke({'num1':4,'num2':2})

print(result)