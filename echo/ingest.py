import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_PATH = "vectorstore/echo"

def load_document(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

def ingest_documents(file_paths: list[str]):
    all_docs = []
    for path in file_paths:
        docs = load_document(path)
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return len(chunks)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

def ingest_data_folder(folder_path: str = "data"):
    supported = [".pdf", ".txt", ".docx"]
    file_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[-1].lower() in supported
    ]
    if not file_paths:
        raise ValueError(f"No supported documents found in {folder_path}/")
    
    print(f"Found {len(file_paths)} documents to ingest:")
    for p in file_paths:
        print(f"  → {p}")
    
    return ingest_documents(file_paths)