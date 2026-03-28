from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from agent import RAGAgent

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = RAGAgent()
    yield

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Chill bro"}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://f1stackmind.vercel.app",
                   "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str
    
def get_agent():
    return agent

@app.post("/messages")
@limiter.limit("5/minute")
async def messages(
    request: Request,
    question: Question,
    agent: RAGAgent = Depends(get_agent)
):
    
    message = agent.ask(question.question)
    
    return message