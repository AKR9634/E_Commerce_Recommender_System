FROM python:3.11-slim

WORKDIR /app

COPY FRONTEND/requirements.txt .

RUN pip install -r requirements.txt

COPY FRONTEND/ ./FRONTEND/

EXPOSE 8501

CMD ["streamlit", "run", "FRONTEND/APP.py", "--server.port=8501", "--server.address=0.0.0.0"]

 