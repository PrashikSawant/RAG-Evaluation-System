import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv
import fitz
import os
import hashlib

load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="rag_eval",
    metadata={"hnsw:space": "cosine"}
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

def extract_text(uploaded_file):
    filename = uploaded_file.name
    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif filename.endswith(".pdf"):
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "".join(page.get_text() for page in doc)
    return ""

def get_doc_id(filename):
    return hashlib.md5(filename.encode()).hexdigest()[:8]

def document_exists(filename):
    results = collection.get(where={"filename": filename})
    return len(results["ids"]) > 0

def add_document(uploaded_file):
    filename = uploaded_file.name
    if document_exists(filename):
        return f"'{filename}' already exists."
    text = extract_text(uploaded_file)
    if not text.strip():
        return "Could not extract text."
    chunks = splitter.split_text(text)
    embeddings = embedding_model.encode(chunks).tolist()
    doc_id = get_doc_id(filename)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"filename": filename, "chunk": i}
                   for i in range(len(chunks))]
    )
    return f"✅ '{filename}' added — {len(chunks)} chunks stored."

def get_documents_list():
    if collection.count() == 0:
        return []
    results = collection.get()
    return list(set(m["filename"] for m in results["metadatas"]))

def delete_document(filename):
    results = collection.get(where={"filename": filename})
    if results["ids"]:
        collection.delete(ids=results["ids"])
        return f"✅ '{filename}' deleted."
    return "Document not found."

def semantic_search(query, top_k=4):
    if collection.count() == 0:
        return []
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return list(zip(docs, metadatas, distances))

def generate_answer(query, results):
    if not results:
        return "No relevant content found.", ""
    context = "\n\n".join([doc for doc, _, _ in results])
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3
    )
    return response.choices[0].message.content, context