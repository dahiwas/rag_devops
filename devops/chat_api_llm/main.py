from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from utils import ask_gemini
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Carrega variáveis de ambiente
load_dotenv()

app = FastAPI(
    title="API Chat LLM",
    description="API para processamento de texto com Gemini",
    version="1.0.0"
)

# Modelo de embedding
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

class QuestionRequest(BaseModel):
    question: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class QuestionResponse(BaseModel):
    answer: str

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Faz uma pergunta ao modelo Gemini e retorna a resposta.
    Args:
        request (QuestionRequest): Objeto contendo a pergunta e parâmetros opcionais
    Returns:
        QuestionResponse: Resposta do modelo
    """
    try:
        answer = await ask_gemini(
            question=request.question,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return QuestionResponse(answer=answer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar a pergunta: {str(e)}"
        )

@app.post("/generate-embedding", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    """Gera embedding para um texto usando o Gemini"""
    try:
        embedding = embeddings.embed_query(request.text)
        return EmbeddingResponse(embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Verifica a saúde da API"""
    try:
        # Testa se a chave da API está configurada
        if not os.getenv("GEMINI_API_KEY"):
            return {"status": "unhealthy", "error": "GEMINI_API_KEY não configurada"}
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
