curl -X POST "http://localhost:8000/create-collection"

curl -X POST "http://localhost:8000/upload-pdf" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/home/ubuntu/rag/Acne.pdf"

curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "o que é acne",
       "limit": 5
     }'

