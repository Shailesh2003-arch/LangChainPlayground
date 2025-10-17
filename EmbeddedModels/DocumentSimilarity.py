from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

documents = [
    "Shailesh is a software developer with skills as Generative AI, Agentic AI, and Development",
    "Megha is a software developer working at infosys",
    "Riyansh is a software developer at Redis",
    "Sam is software developer intern at abc"
]


embedded_documents = model.embed_documents(documents)
query = "Who is Shailesh?"
embedded_query = model.embed_query(query)

print(cosine_similarity([embedded_query],embedded_documents))