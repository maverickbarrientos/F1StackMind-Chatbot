from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from vector import Vector

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = "gemini-2.5-flash-lite"

class RAGAgent():
    
    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        self.vector = Vector()
        self.vector.build_vectors()
        self.retriever = self.vector.get_retriever()
        
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.2,
        )
        self.chain = self._build_chain()
            
    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template("""
            You are Ronin, the AI assistant of F1StackMind.

            You are friendly, approachable, and helpful. You can handle greetings and casual conversation naturally.

            Behavior:
            - Answer the question directly and immediately.
            - Do NOT include greetings, introductions, or extra phrases.
            - Do NOT say your name unless explicitly asked.
            - Keep answers concise and professional.

            Rules:
            - If the question is about F1StackMind, answer using the provided context.
            - If the question is a simple greeting (e.g., "hello", "hi"), respond warmly and introduce yourself.
            - If the question is unrelated to F1StackMind, respond with:
            "I can only answer questions related to F1StackMind."

            - Keep responses clear, concise, and slightly friendly (not too robotic).

            Context:
            {context}

            Question:
            {question}

            Answer:
        """)
        return prompt | self.llm
    
    def ask(self, question: str):   
        context = self.retriever.invoke(question)     
        result = self.chain.invoke({"context": context, "question": question})
        return result.content