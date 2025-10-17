import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# get the api key from environment variable.
API_KEY = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY, temperature=1)

result = model.invoke("Write something about a good software developer in 5 lines")

print(result.content)
