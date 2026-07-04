import base64

from fastembed import TextEmbedding
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

model = TextEmbedding("BAAI/bge-small-en-v1.5") 

app = FastAPI()


class Req(BaseModel):
    query: str


@app.post("/embed")
def embed(req: Req):
    vec = next(iter(model.embed([req.query])))
    raw = vec.astype("<f4").tobytes()
    return {"embedding_b64": base64.b64encode(raw).decode()}


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100)
