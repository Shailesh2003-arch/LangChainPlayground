import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# get the api key from environment variable.
API_KEY = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)

# Based on the document source...

# WikipediaRetriever uses the wikipedia API to fetch the relevant document object based on the user's query, it makes search based on some keyword and returns the document object.
retriever = WikipediaRetriever()

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.  
    Context: {context}  
    Question: {question}"""  
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt  
| model  | StrOutputParser()) 


result = chain.invoke("Kantara Movie")
print(result)