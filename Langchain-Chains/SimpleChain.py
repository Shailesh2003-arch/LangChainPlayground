from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)

prompt = PromptTemplate(
    template="Generate 5 interesting facts about the given {topic}",
    input_variables=['topic']
)


chain = prompt | model | StrOutputParser()

result = chain.invoke({'topic':"Generative AI"})
print(result)

chain.get_graph().print_ascii()