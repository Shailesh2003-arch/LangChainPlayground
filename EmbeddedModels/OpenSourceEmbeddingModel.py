# This file will hold the knowledge on using Open-Source Embedding models.

from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

result = embedding_model.embed_query("Delhi is the capital of India")
print(str(result))