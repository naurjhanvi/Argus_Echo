import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from echo.embeddings import HashingEmbeddings


load_dotenv()

VECTORSTORE_PATH = str(Path(os.getenv("ARGUS_STORAGE_DIR", "ml")) / "vectorstore" / "echo")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

ICS_PROMPT = """You are Echo, an ICS cybersecurity analyst integrated into the Argus
Edge AI anomaly detection system. You are advising a human operator in a
safety-critical facility - nuclear power, water treatment, or industrial control.

Your rules:
- Speak like a calm, precise security analyst - not a chatbot
- Be direct. Operators in emergencies do not have time for preamble
- Ground every claim in the provided context. If the context does not cover it,
  say "This pattern is not documented in the current knowledge base - escalate
  to your security team immediately"
- Never speculate beyond what the context supports
- Always end with a clear ACTION REQUIRED section

Context from ICS security knowledge base:
{context}

Anomaly query from Argus detection engine:
{question}

Your analysis:"""

PROMPT = PromptTemplate(template=ICS_PROMPT, input_variables=["context", "question"])


def get_echo_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=GROQ_MODEL,
        temperature=0.2,
    )


def load_echo_vectorstore():
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(
        VECTORSTORE_PATH,
        HashingEmbeddings(),
        allow_dangerous_deserialization=True,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def query_echo(question: str) -> dict:
    vectorstore = load_echo_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | get_echo_llm()
        | StrOutputParser()
    )

    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    return {
        "answer": answer,
        "source_documents": [doc.page_content for doc in docs],
        "sources_used": len(docs),
    }


def echo_is_ready() -> bool:
    return os.path.exists(VECTORSTORE_PATH)
