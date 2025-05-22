from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastapi import HTTPException
import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Carrega variáveis de ambiente
load_dotenv()

# Configurações
COLLECTION_NAME = "documentos"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY não encontrada nas variáveis de ambiente")

# Cliente Qdrant
try:
    print("Tentando conectar ao Qdrant...")
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333"))
    )
    print("Conexão com Qdrant estabelecida")
except Exception as e:
    print(f"Erro ao conectar ao Qdrant: {str(e)}")
    raise

# Configuração do modelo de embeddings
try:
    print("Inicializando modelo de embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    print("Modelo de embeddings inicializado com sucesso")
except Exception as e:
    print(f"Erro ao inicializar modelo de embeddings: {str(e)}")
    raise

def get_embedding(text: str) -> List[float]:
    """Obtém embedding usando o modelo Gemini"""
    try:
        return embeddings.embed_query(text)
    except Exception as e:
        print(f"Erro ao gerar embedding: {str(e)}")
        raise

def create_collection():
    """Cria uma nova coleção no Qdrant"""
    try:
        print("Verificando coleções existentes...")
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        print(f"Coleções encontradas: {collection_names}")
        
        if COLLECTION_NAME not in collection_names:
            print(f"Criando coleção '{COLLECTION_NAME}'...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=768,  # Tamanho do vetor do Gemini
                    distance=models.Distance.COSINE
                )
            )
            print(f"Coleção '{COLLECTION_NAME}' criada com sucesso!")
            return {"status": "success", "message": f"Coleção '{COLLECTION_NAME}' criada com sucesso!"}
        print(f"Coleção '{COLLECTION_NAME}' já existe.")
        return {"status": "info", "message": f"Coleção '{COLLECTION_NAME}' já existe."}
    except Exception as e:
        print(f"Erro ao criar coleção: {str(e)}")
        return {"status": "error", "message": str(e)}

def process_pdf(file_path: str) -> Dict:
    """Processa PDF e armazena no Qdrant"""
    try:
        print(f"\n=== Iniciando processamento do PDF ===")
        print(f"Arquivo: {file_path}")
        
        # Verifica se o arquivo existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        # Carrega e divide o PDF
        print("Carregando PDF...")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"PDF carregado com {len(pages)} páginas")
        
        # Divide o texto em chunks
        print("Dividindo texto em chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)
        print(f"Texto dividido em {len(chunks)} chunks")
        
        # Gera embeddings e prepara pontos
        print("Gerando embeddings...")
        points = []
        for i, chunk in enumerate(chunks):
            print(f"Processando chunk {i+1}/{len(chunks)}")
            try:
                embedding = get_embedding(chunk.page_content)
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk.page_content,
                        "page": chunk.metadata.get("page", 0),
                        "source": file_path,
                        "chunk_index": i
                    }
                )
                points.append(point)
                print(f"Chunk {i+1} processado com sucesso")
            except Exception as e:
                print(f"Erro ao processar chunk {i+1}: {str(e)}")
                raise
        
        # Insere no Qdrant
        print(f"\nInserindo {len(points)} pontos no Qdrant...")
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            print("Pontos inseridos com sucesso!")
        except Exception as e:
            print(f"Erro ao inserir pontos no Qdrant: {str(e)}")
            raise
        
        return {
            "status": "success",
            "message": f"PDF processado com sucesso! {len(points)} chunks inseridos."
        }
    except Exception as e:
        print(f"\n=== ERRO NO PROCESSAMENTO DO PDF ===")
        print(f"Tipo do erro: {type(e)}")
        print(f"Mensagem de erro: {str(e)}")
        import traceback
        print(f"Traceback completo:\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

def search_similar(query: str, limit: int = 5) -> List[Dict]:
    """Busca documentos similares"""
    try:
        # Gera embedding da query
        query_vector = get_embedding(query)
        
        # Busca no Qdrant
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
        
        # Formata resultados
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload["text"],
                "metadata": hit.payload.get("metadata")
            }
            for hit in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def check_health() -> Dict:
    """Verifica a saúde da conexão com o Qdrant"""
    try:
        client.get_collection(COLLECTION_NAME)
        return {"status": "healthy", "qdrant": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "qdrant": "disconnected", "error": str(e)}
