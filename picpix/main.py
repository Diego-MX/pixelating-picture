import sys
from fastapi import FastAPI
import uvicorn 

debug = 'debug' in sys.args

app = FastAPI()

@app.get("/")
async def root():
    return {'message': "Hello World"}



if __name__ == '__main__': 
    uvicorn.run("main:app", host="127.0.0.1", port=80, debug=debug)