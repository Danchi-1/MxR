from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile):
    # TODO: implement upload handling
    return {"message": "Upload endpoint not yet implemented"}

@app.post("/prompt")
async def prompt(payload: dict):
    # TODO: implement prompt handling
    return {"message": "Prompt endpoint not yet implemented"}

@app.post("/rechunk")
async def rechunk(payload: dict):
    # TODO: implement rechunk handling
    return {"message": "Rechunk endpoint not yet implemented"}