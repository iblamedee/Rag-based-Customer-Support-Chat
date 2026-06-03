from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query 
from queues.worker import process_queue
from client.queue import queue


app = FastAPI()


@app.get("/")
def home():
    return {"message": "Welcome to the RAG Queue Service"}


@app.post("/chat")
def chat(query: str= Query(..., description="The user's question to be processed")
    ):
    
    job = queue.enqueue(process_queue, query)


    return {"status": "Your query is being processed", "job_id": job.id}

@app.get("/result")
def get_result(job_id: str = Query(..., description=" this is user job ID")):

    job= queue.fetch_job(job_id = job_id)
    result = job.return_value()


    return {"result": result}