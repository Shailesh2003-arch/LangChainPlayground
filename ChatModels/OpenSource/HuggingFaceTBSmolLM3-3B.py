from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    task="text-generation",
)


model = ChatHuggingFace(llm=llm)


result = model.invoke("Who is the President of India? in one line.")
print(result)