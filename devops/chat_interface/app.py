import streamlit as st
import httpx
import os
import asyncio
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="Chat Interface",
    page_icon="🤖",
    layout="wide"
)

# Configuração das APIs
QDRANT_API_URL = os.getenv("QDRANT_API_URL", "http://api-chat-bd:8000")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "http://chat-api-llm:8000")

# Função para executar código assíncrono
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Função para criar coleção
def create_collection():
    try:
        response = httpx.post(f"{QDRANT_API_URL}/create-collection")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erro ao criar coleção: {str(e)}")
        return None

# Função para fazer upload de PDF
def upload_pdf(file):
    try:
        files = {"file": file}
        response = httpx.post(f"{QDRANT_API_URL}/upload-pdf", files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erro ao fazer upload do PDF: {str(e)}")
        return None

# Função para buscar documentos
def search_documents(query, limit=5):
    try:
        response = httpx.post(
            f"{QDRANT_API_URL}/search",
            json={"query": query, "limit": limit}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erro ao buscar documentos: {str(e)}")
        return []

# Função assíncrona para chat com Gemini
async def chat_with_gemini(message, history):
    try:
        data = {
            "question": message,
            "temperature": 0.7
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{GEMINI_API_URL}/ask", json=data)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Erro no chat com Gemini: {str(e)}")
        return None

# Sidebar para upload de PDFs
with st.sidebar:
    st.title("Upload de PDFs")
    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processando PDF..."):
            result = upload_pdf(uploaded_file)
            if result and result.get("status") == "success":
                st.success("PDF processado com sucesso!")
            else:
                st.error("Erro ao processar PDF")

# Tabs principais
tab1, tab2 = st.tabs(["Busca de Documentos", "Chat"])

# Tab de Busca
with tab1:
    st.title("Busca de Documentos")
    
    # Botão para criar coleção
    if st.button("Criar Coleção"):
        with st.spinner("Criando coleção..."):
            result = create_collection()
            if result and result.get("status") in ["success", "info"]:
                st.success(result.get("message", "Coleção criada com sucesso!"))
            else:
                st.error("Erro ao criar coleção")
    
    # Input para busca
    query = st.text_input("Digite sua busca:")
    limit = st.slider("Número de resultados", 1, 10, 5)
    
    if query:
        with st.spinner("Buscando..."):
            results = search_documents(query, limit)
            if results:
                for i, result in enumerate(results, 1):
                    with st.expander(f"Resultado {i} (Score: {result['score']:.2f})"):
                        st.write(result["text"])
                        if result.get("metadata"):
                            st.write("Metadados:", result["metadata"])
            else:
                st.info("Nenhum resultado encontrado")

# Tab de Chat
with tab2:
    st.title("Chat")
    
    # Container para o input do chat
    chat_input_container = st.container()
    
    # Container para o histórico de chat
    chat_history_container = st.container()
    
    # Inicializa o histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Input do usuário (agora no topo)
    with chat_input_container:
        if prompt := st.chat_input("Digite sua mensagem"):
            # Adiciona a mensagem do usuário ao histórico
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Busca documentos relevantes
            with st.spinner("Buscando informações relevantes..."):
                results = search_documents(prompt)
                context = "\n".join([r["text"] for r in results]) if results else ""
            
            # Gera resposta usando Gemini
            with st.spinner("Gerando resposta com Gemini..."):
                if context:
                    # Prepara o prompt para o Gemini
                    gemini_prompt = f"""Com base no seguinte contexto, responda à pergunta do usuário de forma clara e concisa.
                    Se a resposta não estiver no contexto, diga que não tem informações suficientes.

                    Contexto:
                    {context}

                    Pergunta do usuário: {prompt}

                    Resposta:"""
                    
                    # Obtém resposta do Gemini
                    gemini_response = run_async(chat_with_gemini(gemini_prompt, []))
                    
                    if gemini_response and "answer" in gemini_response:
                        response = gemini_response["answer"]
                    else:
                        response = "Desculpe, não consegui gerar uma resposta adequada com o Gemini."
                else:
                    response = "Não encontrei informações relevantes nos documentos para responder sua pergunta."
            
            # Adiciona a resposta ao histórico
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Exibe o histórico de chat (agora abaixo do input)
    with chat_history_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])