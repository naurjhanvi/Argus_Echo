import os
from pathlib import Path

from pypdf import PdfReader

from echo.embeddings import HashingEmbeddings


VECTORSTORE_PATH = str(Path(os.getenv("ARGUS_STORAGE_DIR", "ml")) / "vectorstore" / "echo")


def load_document_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".txt":
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    if ext == ".docx":
        import docx2txt

        return docx2txt.process(file_path) or ""
    if ext == ".inp":
        # EPANET INP is already a text format. Prefix each record with its
        # section so retrieved network entries retain their meaning in RAG.
        raw_text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        current_section = "EPANET NETWORK MODEL"
        readable_lines = []
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line.strip("[] ").upper()
                readable_lines.append(f"Section: {current_section}")
            else:
                readable_lines.append(f"{current_section}: {line}")
        return "\n".join(readable_lines)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


def ingest_documents(file_paths: list[str]):
    from langchain_community.vectorstores import FAISS

    texts = []
    metadatas = []
    for path in file_paths:
        for index, chunk in enumerate(chunk_text(load_document_text(path))):
            texts.append(chunk)
            metadatas.append({"source": Path(path).name, "source_path": str(Path(path)), "chunk": index})

    if not texts:
        raise ValueError("No text could be extracted from the supplied documents.")

    vectorstore = FAISS.from_texts(texts, HashingEmbeddings(), metadatas=metadatas)
    vectorstore.save_local(VECTORSTORE_PATH)
    return len(texts)


def load_vectorstore():
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(
        VECTORSTORE_PATH,
        HashingEmbeddings(),
        allow_dangerous_deserialization=True,
    )


def ingest_data_folder(folder_path: str = "data"):
    supported = [".pdf", ".txt", ".docx", ".inp"]
    file_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[-1].lower() in supported
    ]
    if not file_paths:
        raise ValueError(f"No supported documents found in {folder_path}/")

    print(f"Found {len(file_paths)} documents to ingest:")
    for path in file_paths:
        print(f"  -> {path}")

    return ingest_documents(file_paths)
