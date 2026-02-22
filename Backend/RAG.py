text = """
This is my long document...
Add your real data (website text, PDF extracted text, JSON, reports, etc.)
It will get chunked and stored in FAISS.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5,
    chunk_overlap=2
)

chunks = text_splitter.split_text(text)

print("Total Chunks Created:", len(chunks))

embeddings = OllamaEmbeddings(
    model="llama3.2:1b",
)

vector_store = FAISS.from_texts(chunks, embedding=embeddings)

print("FAISS index created.")

vector_store.save_local("faiss_index")
print("FAISS index saved to folder: faiss_index/")

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded back.")

query = "What is this document about?"

results = vector_store.similarity_search(query, k=3)

print("\n---- SEARCH RESULTS ----")
for idx, r in enumerate(results):
    print(f"\nChunk {idx+1}:")
    print(r.page_content)