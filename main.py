from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import ask_question

app = FastAPI()

class Query(BaseModel):
    query: str


@app.post("/ask")
def ask(data: Query):

    answer = ask_question(data.query)

    return {"answer": answer}
