import uvicorn
import os
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel, ValidationError
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any

# Import the QA service AND the DataFrame (df)
from app.services.qa_service import get_answer_from_data, df

app = FastAPI(title="Kearney AI Chatbot API")

# --- 1. CORS Middleware ---
# We allow all origins for the demo
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Request/Response Models ---
class ChatMessage(BaseModel):
    sender: str
    text: str

class ChatRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = None

class ChartData(BaseModel):
    type: str
    labels: List[str]
    data: List[float]

class ChatResponse(BaseModel):
    answer: str
    chart: Optional[ChartData] = None

# --- 3. API Endpoints ---
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
async def handle_chat(request: Request): # Changed to use raw Request
    """
    Receives a chat history, passes it to the QA service,
    and returns the answer.
    """
    
    # --- FIX: Manually handle body parsing to avoid OPTIONS error ---
    try:
        body = await request.json()
        chat_request = ChatRequest(**body)
    except Exception as e:
        # This catches empty bodies from OPTIONS and other malformed requests
        print(f"Could not parse request body: {e}")
        return ChatResponse(answer="Invalid request format.", chart=None)
    
    if not chat_request or not chat_request.messages:
        return ChatResponse(answer="No query provided.", chart=None)

    history_messages = [msg.dict() for msg in chat_request.messages[:-1]]
    current_query = chat_request.messages[-1].text
    
    result = {'answer': 'An error occurred.', 'chart': None}
    try:
        # This function now returns a dict: {'answer': ..., 'chart': ...}
        result = get_answer_from_data(current_query, history_messages)
    except Exception as e:
        print(f"Error in get_answer_from_data: {e}")
        result = {'answer': f'An internal error occurred: {e}', 'chart': None}

    # Safely validate the chart data before returning
    validated_chart = None
    if result.get('chart'):
        try:
            # Try to parse the chart data into our pydantic model
            validated_chart = ChartData(**result['chart'])
        except Exception:
            # If validation fails, just set chart to None
            validated_chart = None
    
    return ChatResponse(answer=result['answer'], chart=validated_chart)

# --- 4. Run the App ---
if __name__ == "__main__":
    uvicorn.run("app.main:app", 
                host="127.0.0.1", 
                port=8000, 
                reload=True
               )

