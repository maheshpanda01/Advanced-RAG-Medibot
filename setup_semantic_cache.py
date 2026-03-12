from langchain.cache import  RedisSemanticCache
from langchain.globals import set_llm_cache
from langchain_openai import OpenAIEmbeddings


semantic_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    score_threshold=0.85

)

set_llm_cache(semantic_cache)
