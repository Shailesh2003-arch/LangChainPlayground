from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

parser = JsonOutputParser()


llm = HuggingFaceEndpoint(
    # repo_id="HuggingFaceTB/SmolLM3-3B",
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)



template = PromptTemplate(
    template="Give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format()

chain = template | model | parser

result = chain.invoke({})
print(result)


# via using JSONOutputParser we can bring in the json object but we cannot enforce schema validation on it!...