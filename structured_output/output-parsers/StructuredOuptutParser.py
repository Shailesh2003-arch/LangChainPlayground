from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core import StructuredOutputParser, ResponseSchema
# from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema


"""
StructuredOutputParser was part of the old LangChain architecture (versions around 0.0.310 to 0.1.x).
When LangChain 1.0.0+ dropped (especially in early 2024), the devs massively refactored the library — splitting it into multiple modular packages and moving many utilities to new namespaces.
"""



load_dotenv()


llm = HuggingFaceEndpoint(
    # repo_id="HuggingFaceTB/SmolLM3-3B",
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# You need to make a schema that will guide LLMs response.

schema = [
    ResponseSchema(name='fact_1', description="Fact 1 about the topic"),
     ResponseSchema(name='fact_2', description="Fact 2 about the topic"),
      ResponseSchema(name='fact_3', description="Fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template="Give 3 fact about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


prompt = template.invoke({'topic':"Generative AI"})
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)