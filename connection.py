from fastapi import FastAPI
from pydantic import BaseModel
import torch
from testplangeneration import model, tokenizer    

app = FastAPI()

class Request(BaseModel):
    instruction: str

@app.post("/generate")
def generate(req: Request):
    prompt = f"""### Instruction:
    {req.instruction}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.2
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}