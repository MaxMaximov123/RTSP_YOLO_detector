import asyncio
import threading
import cv2
import numpy as np
from datetime import datetime
from aiohttp import web
from ultralytics import YOLO
import websockets
import torch
from collections import deque
import time
import os
import io
import telebot
from dotenv import load_dotenv

load_dotenv()

# === Системные настройки ===
os.environ["GLOG_minloglevel"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# print("CUDA available:", torch.cuda.is_available())

# === Настройки ===
RTSP_URL = os.getenv("RTSP_URL")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))

model = YOLO("yolo11s.pt")

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)
last_sent_time = 0

# === Глобальные переменные ===
latest_frame = None
frame_lock = threading.Lock()

inference_queue = deque(maxlen=1)
processed_frame = None
processed_timestamp = None
processed_lock = threading.Lock()

# === Чтение RTSP ===
def rtsp_reader():
    global latest_frame
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("❌ Не удалось открыть RTSP")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame.copy()
            if len(inference_queue) == 1:
                inference_queue.popleft()
            inference_queue.append(latest_frame.copy())

# === Инференс + отправка в Telegram ===
def inference_worker():
    global processed_frame, processed_timestamp, last_sent_time
    while True:
        if len(inference_queue) == 0:
            time.sleep(0.001)
            continue
        frame = inference_queue.popleft()

        results = model.predict(
            frame,
            verbose=False,
            stream=True,
            imgsz=416,
            conf=0.3,
            classes=[0, 15, 16]
        )

        result = next(results)
        objects_found = False

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            objects_found = True

        with processed_lock:
            processed_frame = frame
            processed_timestamp = datetime.now()

        # Отправка в Telegram
        now = time.time()
        if objects_found and (now - last_sent_time > 3):
            last_sent_time = now
            _, buffer = cv2.imencode('.jpg', frame)
            img_bytes = buffer.tobytes()
            send_to_telegram(img_bytes)

# === Отправка в Telegram через поток ===
def send_to_telegram(image_bytes: bytes):
    def worker():
        try:
            image_io = io.BytesIO(image_bytes)
            image_io.name = "detected.jpg"
            bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=image_io, caption="Объект(ы) обнаружены")
        except Exception as e:
            print(f"❌ Ошибка при отправке в Telegram: {e}")

    threading.Thread(target=worker, daemon=True).start()

# === WebSocket обработка ===
async def ws_handler(ws):
    global processed_frame, processed_timestamp
    print("🔌 WebSocket клиент подключён")

    prev_frame_time = None
    fps = 0.0

    try:
        while True:
            await asyncio.sleep(0.001)

            with processed_lock:
                if processed_frame is None or processed_timestamp is None:
                    continue
                frame = processed_frame.copy()
                timestamp = processed_timestamp

            if prev_frame_time:
                elapsed = (timestamp - prev_frame_time).total_seconds()
                if elapsed > 0:
                    fps = 1.0 / elapsed
            prev_frame_time = timestamp

            h, w = frame.shape[:2]
            time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, time_str, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            _, jpeg = cv2.imencode('.jpg', frame)
            await ws.send(jpeg.tobytes())

    except websockets.exceptions.ConnectionClosed:
        print("❌ WebSocket отключён")

# === WebSocket сервер ===
async def start_ws():
    async with websockets.serve(ws_handler, "0.0.0.0", 8765, max_size=2**22):
        await asyncio.Future()

# === HTTP сервер для отдачи index.html ===
async def start_http():
    app = web.Application()
    app.router.add_static("/", path="static", show_index=True)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 3000)
    await site.start()
    print("🌍 HTTP сервер на http://localhost:3000")

# === Основной async запуск ===
async def main():
    threading.Thread(target=rtsp_reader, daemon=True).start()
    threading.Thread(target=inference_worker, daemon=True).start()
    await asyncio.gather(start_ws(), start_http())

if __name__ == "__main__":
    asyncio.run(main())