from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def create_hybrid_retriever(db):

    # Semantic retriever
    faiss_retriever = db.as_retriever(search_kwargs={"k": 5})

    # Keyword retriever
    documents = list(db.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    # Hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    return hybrid_retriever
