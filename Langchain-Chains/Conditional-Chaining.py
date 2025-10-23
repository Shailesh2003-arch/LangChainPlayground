# This is a conditional chaining, which only executes one chain at a time.
# Based on the condition.

from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

parser = StrOutputParser()

# Here this pydantic model will ensure strict structured output.
class Feedback(BaseModel):
    sentiment: Literal['Positive','Negative']


pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into Positive or Negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':pydantic_parser.get_format_instructions()}
)


classifier_chain = prompt1 | model | pydantic_parser

prompt2 = PromptTemplate(
    template="Create an appropriate response for this Positive feedback text in 3 lines dont provide any optional content\n {feedbacktext}",
    input_variables=['feedbacktext']
)

prompt3 = PromptTemplate(
    template="Create an appropriate response for this Negative feedback in 3 lines dont provide any optional content \n {feedbacktext}",
    input_variables=['feedbacktext']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment =='Positive', prompt2 | model | parser),
    (lambda x:x.sentiment =='Negative', prompt3 | model | parser),
    lambda x:"Goodbye!"
)

# result = classifier_chain.invoke({'feedback':"This is a terrible smartphohne."}).sentiment
# print(result)

chain = classifier_chain | branch_chain
result = chain.invoke({'feedback':"This is a terrible smartphohne."})
print(result)