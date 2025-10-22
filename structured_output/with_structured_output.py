from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from pydantic import BaseModel

load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=API_KEY)



# You need to define a schema...
class Review(TypedDict):
    summary:str
    sentiment:str



# structured_model = model.with_structured_output(Review)

# result = structured_model.invoke("""
# The hardware is great, but the software feels bloated. There are too many pre-installed apps which I can't uninstall. Also the UI looks outdated compared to other brands. Hoping for a software update to fix this. 
# """)

# print(result)


# Another examples using Pydantic library's schema validation.
class Movie(BaseModel):
    title: str
    director: str
    year: int

structured_model = model.with_structured_output(Movie,method="json_mode")

result = structured_model.invoke("""
Tell me about the movie interstellar 
""")

print(result.model_dump_json(indent=2))
# print(type(result))