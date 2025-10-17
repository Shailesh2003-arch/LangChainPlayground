from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings

load_dotenv()

API_KEY = os.getenv("OPEN_AI_API_KEY") 

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32,api_key=API_KEY)

result = embedding_model.embed_query("Delhi is the capital of India")

print(str(result))