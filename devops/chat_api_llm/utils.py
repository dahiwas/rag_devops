import os
import logging
from dotenv import load_dotenv
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

async def ask_gemini(
    question: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Função simplificada para fazer perguntas ao Gemini usando LangChain.
    
    Args:
        question (str): A pergunta a ser feita
        temperature (float): Temperatura para geração
        max_tokens (int, optional): Número máximo de tokens
        
    Returns:
        str: Resposta do modelo
    """
    try:
        # Inicializa o modelo Gemini
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Cria a mensagem
        messages = [HumanMessage(content=question)]
        
        # Gera a resposta
        logger.info(f"Enviando pergunta para o Gemini: {question[:100]}...")
        response = model.invoke(messages)
        logger.info("Resposta recebida com sucesso")
        
        return response.content
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {str(e)}")
        raise

# Exemplo de uso
if __name__ == "__main__":
    import asyncio
    
    async def main():
        try:
            # Exemplo com diferentes configurações
            resposta = await ask_gemini(
                "Explique o que é inteligência artificial em 3 linhas.",
                temperature=0.5,
                max_tokens=150
            )
            print(f"\nResposta do Gemini:\n{resposta}")
            
        except Exception as e:
            print(f"\nErro ao fazer a requisição: {str(e)}")
    
    asyncio.run(main())
