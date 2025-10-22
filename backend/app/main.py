import uvicorn
import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any

from app.services.qa_service import get_answer_from_data, df

app = FastAPI(title="Kearney AI Chatbot API")

origins = [
    "http://localhost:3000",
    "localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    sender: str
    text: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChartData(BaseModel):
    type: str
    labels: List[str]
    data: List[float]

class ChatResponse(BaseModel):
    answer: str
    chart: Optional[ChartData] = None

@app.get("/")
def read_root():
    return {"status": "Kearney AI Chatbot API is running"}

@app.get("/api/data", response_model=List[Dict[str, Any]])
async def get_raw_data():
    """
    Fetches the entire DataFrame and returns it as JSON records.
    """
    data_json = json.loads(df.to_json(orient='records'))
    return data_json

@app.post("/api/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Receives a chat history, passes it to the QA service,
    and returns the answer.
    """
    if not request.messages:
        return ChatResponse(answer="No query provided.")

    history_messages = [msg.dict() for msg in request.messages[:-1]]
    current_query = request.messages[-1].text
    
    result = get_answer_from_data(current_query, history_messages)
    
    validated_chart = None
    raw_chart = result.get('chart')

    if raw_chart:
        try:
            validated_chart = ChartData.model_validate(raw_chart)
        except Exception:
            validated_chart = None
    
    return ChatResponse(answer=result['answer'], chart=validated_chart)

if __name__ == "__main__":
    uvicorn.run("app.main:app", 
                host="127.0.0.1", 
                port=8000, 
                reload=True
               )


