import os
import cv2
import subprocess
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Импортируем наши собственные модули
import core
import database

app = FastAPI()

# Настройки путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")

for folder in [UPLOAD_DIR, RESULT_DIR]:
    os.makedirs(folder, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    # 1. Сохранение входящего файла
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    raw_path = os.path.join(RESULT_DIR, "temp_raw.mp4")
    web_filename = f"web_{datetime.now().strftime('%H%M%S')}_{file.filename.split('.')[0]}.mp4"
    web_path = os.path.join(RESULT_DIR, web_filename)

    # 2. Инициализация видео
    cap = cv2.VideoCapture(temp_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    violation_detected = False
    frame_count = 0

    # 3. Цикл обработки
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Используем модель из модуля core
        results = core.model.track(frame, persist=True, verbose=False, device=core.DEVICE, conf=0.45, imgsz=640)

        persons = []
        skateboards = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                coords = box.xyxy[0].cpu().numpy()
                if cls == 0: persons.append(coords)
                elif cls == 36: skateboards.append(coords)

        used_p = set()
        for s_box in skateboards:
            riding = False
            for i, p_box in enumerate(persons):
                # Вызываем логику проверки из модуля core
                if i not in used_p and core.is_riding(s_box, p_box):
                    riding = True
                    used_p.add(i)
                    break

            x1, y1, x2, y2 = map(int, s_box)
            color = (0, 0, 255) if riding else (0, 255, 0)
            label = "VIOLATION" if riding else "CARRYING"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if riding: violation_detected = True

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # 4. Конвертация FFmpeg
    subprocess.run([
        'ffmpeg', '-y', '-i', raw_path,
        '-vcodec', 'libx264', '-preset', 'ultrafast', '-crf', '28',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', web_path
    ], capture_output=True)

    if os.path.exists(raw_path): os.remove(raw_path)
    if os.path.exists(temp_path): os.remove(temp_path)

    # 5. Сохранение в историю через модуль database
    history_entry = {
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "filename": file.filename,
        "violation": "Да" if violation_detected else "Нет",
        "frames": frame_count
    }
    database.save_to_history(history_entry)

    return {
        "violation": "Yes" if violation_detected else "No",
        "video_url": f"/static/results/{web_filename}"
    }

@app.get("/get_history")
async def get_history():
    return database.get_all_history()

@app.get("/export_excel")
async def export_excel():
    report_name = "report_skate.xlsx"
    report_path = os.path.join(RESULT_DIR, report_name)
    if database.create_excel_report(report_path):
        return FileResponse(report_path, filename="Report_AntiSkate.xlsx")
    return {"error": "История пуста"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)