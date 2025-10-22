# This file is about Output-parser for the LLMs who don't support structured output format by default.
# We need structured Output format for working with different applications.
# Note that - OutputParser work with both models - model that supports structured format and models that doesn't support structured format...

# StrOutputParser -> returns the string format in response from an LLM.
# Mostly used while you're working with chains that deals with text.

from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm = HuggingFaceEndpoint(
    # repo_id="HuggingFaceTB/SmolLM3-3B",
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    name="Detailed Report Generator",
    template="Generate a Detailed Report on the following topic {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    name="Summary Generator",
    template="Generate a 5 lines summary from the given text {text}",
    input_variables=['text']
)


chain = template1 | model | StrOutputParser() | template2 | model | StrOutputParser()
result = chain.invoke({'topic':"Agentic AI"})
print(result)


