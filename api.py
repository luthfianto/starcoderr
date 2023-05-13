from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import uvicorn

app = FastAPI()

API_KEY = ""


def verify_api_key(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoderbase"
device = "cuda" # for GPU usage or "cpu" for CPU usage

use_auth_token="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"
tokenizer = AutoTokenizer.from_pretrained(checkpoint,use_auth_token=use_auth_token)
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True,use_auth_token=use_auth_token).to(device)

class MyBody(BaseModel):
    text: str ="def print_hello_world():"

@app.post("/chat/")
async def create_chat(
    body: MyBody, api_key_valid: bool = Depends(verify_api_key)
):
    inputs = tokenizer.encode(body.text, return_tensors="pt").to(device)
    return model.generate(inputs)



if __name__ == "__main__":
    uvicorn.run(app, port=8081, host="0.0.0.0")
