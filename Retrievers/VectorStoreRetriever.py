from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


documents = [
    Document(page_content="Chroma is a vector Database"),
    Document(page_content="Embeddings convert text into a higher dimensional vectors"),
    Document(page_content="OpenAi provides powerful embedding model"),
    Document(page_content="Riyansh is a dedicated boy."),
]

embedding_model = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# convert vector store into a retriever.

retriever = vector_store.as_retriever(search_kwargs={"k":1})

query = "Who is Riyansh ?"
result = retriever.invoke(query)

print(result)