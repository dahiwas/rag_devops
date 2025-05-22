from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from utils import create_collection, process_pdf, search_similar, check_health

app = FastAPI(
    title="API Chat BD",
    description="API para gerenciamento de documentos e busca vetorial com Qdrant",
    version="1.0.0"
)

# Configurações
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Modelos Pydantic
class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 5

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: Optional[dict] = None

# Rotas
@app.post("/create-collection")
def create_collection_route():
    """Cria uma nova coleção no Qdrant"""
    result = create_collection()
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return JSONResponse(content=result)

@app.post("/upload-pdf")
async def upload_pdf_route(file: UploadFile = File(...)):
    """Faz upload e processa um PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Apenas PDFs são permitidos")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        # Salva o arquivo
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Processa o PDF
        result = process_pdf(file_path)
        
        # Limpa o arquivo
        os.remove(file_path)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return JSONResponse(content=result)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
def search_route(query: SearchQuery):
    """Busca documentos similares"""
    results = search_similar(query.query, query.limit)
    return [SearchResult(**result) for result in results]

@app.get("/health")
def health_check():
    """Verifica a saúde da API"""
    return check_health()

