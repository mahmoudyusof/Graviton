from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Query(BaseModel):
    msg: str

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/")
def translate(query: Query):
    input_ids = tokenizer(query.msg, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return {"msg": tokenizer.decode(outputs[0], skip_special_tokens=True)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
