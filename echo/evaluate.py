import os
import math
import time
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset

load_dotenv()

def get_evaluator_llm():
    return LangchainLLMWrapper(
        ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
    )

def get_evaluator_embeddings():
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

def evaluate_response(
    question: str,
    answer: str,
    source_documents: list[str]
) -> dict:
    try:
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [source_documents],
            "ground_truth": [answer]
        }

        dataset = Dataset.from_dict(data)

        time.sleep(2)

        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=get_evaluator_llm(),
            embeddings=get_evaluator_embeddings()
        )

        scores = results.to_pandas().iloc[0]

        def clean(val):
            try:
                f = float(val)
                return None if math.isnan(f) else round(f, 3)
            except:
                return None

        return {
            "faithfulness": clean(scores["faithfulness"]),
            "answer_relevancy": clean(scores["answer_relevancy"]),
            "context_precision": clean(scores["context_precision"]),
        }

    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
        }