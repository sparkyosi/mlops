FROM --platform=linux/amd64 python:3.12-alpine

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5500

ENV PORT 5500

# Entry point to dynamically bind to the Render-provided $PORT
CMD ["ash", "-c", "streamlit run main.py --server.port=$PORT"]