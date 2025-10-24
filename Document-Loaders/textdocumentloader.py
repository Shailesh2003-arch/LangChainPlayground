from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader


document_loader = TextLoader('./GenerativeAI.txt','utf-8')

loaded_docs = document_loader.load()
print(loaded_docs)
print(f"This is the content of the document : {loaded_docs[0]}")