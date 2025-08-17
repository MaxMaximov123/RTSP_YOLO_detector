FROM python:3.10-slim

# 1. Установка зависимостей системы
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Копируем код
COPY . /app
WORKDIR /app

EXPOSE 3000 8765

CMD ["python", "main.py"]