from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from setup_semantic_cache import set_llm_cache

from setup_hybrid_retriver import create_hybrid_retriever


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = FAISS.load_local(
    "vectorstore/db-faiss",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Hybrid retriever
hybrid_retriever = create_hybrid_retriever(db)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Compression retriever
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=hybrid_retriever,
    base_compressor=compressor
)


def ask_question(query: str):

    docs = compression_retriever.invoke(query)

    context = "\n\n".join([
        f"Page Content: {doc.page_content}"
        for doc in docs
    ])

    system_prompt = f"""
You are a helpful AI Medical Assistant.
Answer ONLY using the provided context.

Context:
{context}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    response = llm.invoke(messages)

    return response.content
