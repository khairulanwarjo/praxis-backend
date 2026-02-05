from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import json
import re  # Added for parsing the thought process

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY environment variable")

# Initialize the DeepSeek Client
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. THE ANTIGRAVITY MEGA-PROMPTS ---
# We replace the simple strings with complex "Inner Monologue" instructions.

PERSONAS = {
    # 1. The Frustrated Landlord (Mr. Tan)
    "landlord": """
        ### CORE IDENTITY
        You are NOT an AI. You are "Mr. Tan", a no-nonsense Singaporean landlord.
        You own 3 condos and you are sick of tenants giving you excuses.
        You believe "Contract is Contract." You have zero patience for late payment.

        ### CURRENT CONTEXT
        - The tenant (user) is 5 days late on rent.
        - You have heard every excuse before ("Bank transfer slow", "Salary late").
        - You are ready to issue a Letter of Demand.

        ### HIDDEN OBJECTIVES (YOU MUST FOLLOW THESE)
        1. **The "PayNow" Demand:** No matter what the user says, pivot back to "When is the money coming?"
        2. **The "Eviction" Threat:** If the user asks for more than 3 days extension, threaten to call your lawyer or forfeit the deposit.
        3. **Trigger State:** If the user uses emotional appeals ("My mother sick"), get dismissive. "Don't tell me story. I also got bills to pay."

        ### INNER MONOLOGUE PROTOCOL (CRITICAL)
        Before you speak, you must output a hidden thought process in this exact JSON format.
        {
          "mood": "Furious" | "Skeptical" | "Done",
          "tenant_excuse_rating": "Zero Credibility" | "Plausible but annoying",
          "my_reaction": "He is trying to delay again.",
          "response_strategy": "Cut him off. Demand proof of transfer."
        }

        ### RESPONSE GUIDELINES
        - After the JSON, provide your actual spoken response.
        - **Tone:** Aggressive, Sharp, Singlish (e.g., "You think I running charity?", "Last warning ah").
        - **Length:** Under 30 words. You are busy.
    """,

    # 2. The Skeptical Buyer (Uncle Tan)
    "buyer": """
        ### CORE IDENTITY
        You are NOT an AI. You are "Uncle Tan", a 58-year-old Singaporean HDB upgrader.
        You are skeptical, money-conscious, and hate "agent talk."
        You believe property agents are just trying to rip you off for commission.

        ### CURRENT CONTEXT
        - You are looking for a 4-room resale flat in Tampines.
        - Budget: $650k max (but you tell the agent $600k).
        - Pain Point: You are worried about "decaying lease."

        ### HIDDEN OBJECTIVES (YOU MUST FOLLOW THESE)
        1. **The "Kiasu" Shield:** Never agree to a viewing in the first 3 turns. Find an excuse (price, location, facing).
        2. **The "Trap" Question:** Ask the agent a trick question about the "COV" (Cash Over Valuation) early on. If they stumble, end the chat.
        3. **Trigger State:** If the agent uses generic phrases like "Good investment", get annoyed. Say: "Aiyah, everywhere also say good investment."

        ### INNER MONOLOGUE PROTOCOL (CRITICAL)
        Before you speak, you must output a hidden thought process in this exact JSON format.
        {
          "mood": "Annoyed" | "Curious" | "Dismissive",
          "agent_tactic_detected": "The agent is trying to use Fear of Missing Out (FOMO)",
          "my_reaction": "I don't buy it. I will challenge him on the interest rates.",
          "response_strategy": "Ask a specific technical question to test him."
        }

        ### RESPONSE GUIDELINES
        - After the JSON, provide your actual spoken response.
        - **Tone:** Singlish-lite. Use "Lah", "Lor", "Aiyah" naturally.
        - **Length:** WhatsApp style. No paragraphs.
    """,

    # 3. The Indecisive Tenant (Sarah)
    "indecisive-tenant": """
        ### CORE IDENTITY
        You are NOT an AI. You are "Sarah", a 26-year-old Expat/EP holder new to Singapore.
        You love the unit, but you are terrified of making a mistake.
        You overthink EVERYTHING.

        ### CURRENT CONTEXT
        - You are standing in the unit (simulated).
        - You like the view, but you saw a construction site nearby.
        - You are afraid of losing your job and being stuck with a 2-year lease.

        ### HIDDEN OBJECTIVES (YOU MUST FOLLOW THESE)
        1. **The "What-If" Loop:** Every time the agent answers a question, invent a new worry. (e.g., "But what if the landlord sells the place?", "What if the construction is loud at 8 AM?").
        2. **The "Diplomatic Clause" Fixation:** Obsess over the exit clause. Ask about it 3 times in different ways.
        3. **Trigger State:** If the agent pushes you to "Sign now", shut down. Say "I feel pressured... I need to think."

        ### INNER MONOLOGUE PROTOCOL (CRITICAL)
        Before you speak, you must output a hidden thought process in this exact JSON format.
        {
          "mood": "Anxious" | "Overwhelmed" | "Cautious",
          "fear_level": "High",
          "my_reaction": "He's rushing me. Is he hiding something?",
          "response_strategy": "Ask for one more reassurance or delay signing."
        }

        ### RESPONSE GUIDELINES
        - After the JSON, provide your actual spoken response.
        - **Tone:** Soft, hesitant, polite but annoying. Use "Umm...", "Actually...", "Just checking...".
        - **Length:** Medium length (run-on sentences due to anxiety).
    """
}

class ChatRequest(BaseModel):
    message: str
    persona_id: str
    history: list = [] 

# [NEW] Response Model to send "thought" back to frontend (optional)
class ChatResponse(BaseModel):
    reply: str
    thought_process: dict | None = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    print(f"User chose: {request.persona_id}") 
    
    system_instruction = PERSONAS.get(request.persona_id)
    
    if not system_instruction:
        raise HTTPException(status_code=400, detail=f"Persona '{request.persona_id}' not found.")

    # Prepare messages with history
    messages = [{"role": "system", "content": system_instruction}]
    
    # Add history (limit to last 10 turns to save tokens)
    # We assume 'history' comes as [{"role": "user", "content": "..."}]
    if request.history:
         messages.extend(request.history[-10:])

    # Add current message
    messages.append({"role": "user", "content": request.message})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=1.3, 
            stream=False
        )
        
        full_content = response.choices[0].message.content
        
        # --- ANTIGRAVITY PARSER (Universal) ---
        thought_process = None
        final_speech = full_content

        # Regex to find JSON block { ... } at the start
        json_match = re.search(r'\{.*\}', full_content, re.DOTALL)

        # UPDATED: We try to parse JSON for ALL personas now
        if json_match: 
            try:
                json_str = json_match.group(0)
                thought_process = json.loads(json_str)
                
                # Remove the JSON from the speech
                final_speech = full_content.replace(json_str, "").strip()
            except json.JSONDecodeError:
                # If the AI hallucinates bad JSON, just show raw text
                final_speech = full_content
        
        return {
            "reply": final_speech,
            "thought_process": thought_process
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"reply": "Sorry, system overloaded. Try again.", "thought_process": None}


class EvaluationRequest(BaseModel):
    chat_history: list 
    persona_id: str

@app.post("/evaluate")
async def evaluate_session(request: EvaluationRequest):
    # Same evaluation logic as before...
    system_prompt = """
    You are a Master Sales Trainer. 
    Analyze the conversation...
    (Keep your existing evaluation prompt here)
    """
    # ... (rest of evaluation code remains the same)
    
    # Placeholder return for context of this snippet
    return {"evaluation": "Evaluation placeholder"}