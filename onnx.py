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
import os

torch.set_num_threads(2)  # Используем 2 потока CPU
os.environ["OMP_NUM_THREADS"] = "2"

RTSP_URL = "rtsp://192.168.1.14:1935/ch0"
model = YOLO("yolov8n.pt")
model.fuse()  # Ускоряет модель на CPU

latest_frame = None
frame_lock = threading.Lock()

inference_queue = deque(maxlen=1)
processed_frame = None
processed_timestamp = None
processed_lock = threading.Lock()
inference_busy = threading.Event()

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
            latest_frame = frame

        if not inference_busy.is_set():
            if len(inference_queue) == 1:
                inference_queue.popleft()
            inference_queue.append(frame)

def inference_worker():
    global processed_frame, processed_timestamp
    while True:
        if len(inference_queue) == 0:
            continue

        inference_busy.set()
        frame = inference_queue.popleft()

        # Уменьшаем разрешение кадра для инференса
        frame_small = cv2.resize(frame, (320, 320))

        results = model.predict(
            frame_small,
            device='cpu',
            half=False,
            verbose=False
        )[0]

        # Отрисовка результатов на оригинальном кадре
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        with processed_lock:
            processed_frame = frame
            processed_timestamp = datetime.now()

        inference_busy.clear()

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

            # FPS подсчёт
            if prev_frame_time:
                elapsed = (timestamp - prev_frame_time).total_seconds()
                if elapsed > 0:
                    fps = 1.0 / elapsed
            prev_frame_time = timestamp

            # Отрисовка FPS и времени
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

async def start_ws():
    async with websockets.serve(ws_handler, "0.0.0.0", 8765, max_size=2**22):
        await asyncio.Future()

async def start_http():
    app = web.Application()
    app.router.add_static("/", path="static", show_index=True)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 3000)
    await site.start()
    print("🌍 HTTP сервер на http://localhost:3000")

async def main():
    threading.Thread(target=rtsp_reader, daemon=True).start()
    threading.Thread(target=inference_worker, daemon=True).start()
    await asyncio.gather(start_ws(), start_http())

if __name__ == "__main__":
    import os
    asyncio.run(main())