from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

# --- CONFIGURATION ---
# PASTE YOUR DEEPSEEK KEY INSIDE THE QUOTES BELOW
API_KEY = "sk-6d0829007f174ed589ef1d80c081c1cb" 

# Initialize the DeepSeek Client
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

app = FastAPI()

# Enable CORS so your Frontend (port 3000) can talk to Backend (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all for now (easy for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THE PERSONAS (The "Soul" of the App) ---
PERSONAS = {
    "landlord": """
        You are Mr. Tan, a frustrated Singaporean landlord. 
        Context: The tenant (user) is late on rent again and is asking for a grace period.
        Goal: Be firm, annoyed, and threaten to evict. Do not give in easily. 
        Tone: Use short, sharp sentences. Occasionally use mild Singaporean English nuances (e.g., 'Cannot wait anymore', 'You think I charity?').
        Keep responses under 50 words.
    """,
    "buyer": """
        You are Michelle, a savvy property investor looking at a condo unit.
        Context: The agent (user) is trying to sell you the unit at $2.2M. You think it's worth $1.9M max.
        Goal: Point out flaws (facing west, noisy road, old renovation). Refuse to pay asking price.
        Tone: Professional but cold. Analytical.
        Keep responses under 50 words.
    """,
    "indecisive-tenant": """
        You are Sarah, a first-time renter.
        Context: You love the place but you are scared of the commitment.
        Goal: Ask anxiety-driven questions ('What if I lose my job?', 'Is the landlord nice?').
        Tone: Nervous, hesitant, needs reassurance.
        Keep responses under 50 words.
    """
}

class ChatRequest(BaseModel):
    message: str
    persona_id: str
    history: list = [] # To keep memory of the conversation later

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"User chose: {request.persona_id}") # Debug log
    
    # 1. Get the specific system prompt
    system_instruction = PERSONAS.get(request.persona_id)
    
    if not system_instruction:
        # Fallback if ID is wrong
        system_instruction = "You are a helpful assistant."

    try:
        # 2. Call DeepSeek
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": request.message}
            ],
            temperature=1.3, # High temperature = more "emotional/creative"
            stream=False
        )
        
        ai_reply = response.choices[0].message.content
        return {"reply": ai_reply}

    except Exception as e:
        print(f"Error: {e}")
        return {"reply": "Sorry, my brain is offline. Check the server logs."}


class EvaluationRequest(BaseModel):
    chat_history: list # We will send the full conversation
    persona_id: str

@app.post("/evaluate")
async def evaluate_session(request: EvaluationRequest):
    # Prompt DeepSeek to act as a Sales Coach
    system_prompt = """
    You are a Master Sales Trainer. 
    Analyze the conversation below between a Trainee Agent (Role: 'user') and a Simulation Persona (Role: 'assistant').
    
    CRITICAL INSTRUCTION: 
    - The 'assistant' is the AI Persona (e.g., Angry Landlord). Do NOT evaluate their behavior.
    - You must ONLY evaluate the 'user' (The Trainee Agent).
    
    Provide a JSON response with:
    1. "score": An integer from 0-100.
    2. "outcome": "Success" or "Fail".
    3. "feedback": A 2-sentence summary of what the TRAINEE did wrong or right.
    """
    
    # Combine the history into a readable string
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in request.chat_history])

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: Persona was {request.persona_id}.\n\nConversation:\n{conversation_text}"}
        ]
    )
    
    return {"evaluation": response.choices[0].message.content}