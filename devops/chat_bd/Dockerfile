FROM qdrant/qdrant:latest

EXPOSE 6333
EXPOSE 6334

# Volume para persistência de dados
VOLUME ["/qdrant/storage"]

# Configurações de saúde
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:6333/health || exit 1

# Comando padrão
CMD ["./qdrant"]