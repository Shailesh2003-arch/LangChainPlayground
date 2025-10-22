# This is a structured-aware LLMs which knows how to output a result in structured format.
# LLMs like Open-AI's GPT-4 and followed series.
# LLMs like Google's Gemini models gemini-2.5-flash and followed series.
# LLMs like Anthropic's claude models.

# for letting this models output in a structured format we have to use an API known as with_structured_output right before invoking a model.

# We can have different ways to format the data we want to have an output as:
# 1) TypeDict
# 2) Pydantic
# 3) JSON Schema


from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

# new_person: Person={
#     "name":"Shailesh",
#     "age":"22"
# }

new_person = Person(
    {
        'name':"Shailesh",
        'age':"23"
    }
)

print(new_person)