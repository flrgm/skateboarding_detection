import os
import cv2
import torch
import json
import subprocess
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

for folder in [UPLOAD_DIR, RESULT_DIR]:
    os.makedirs(folder, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = YOLO('yolo11m.pt').to(DEVICE)
print(f"--- АКТИВНОЕ УСТРОЙСТВО: {model.device.type} ---")

def is_riding(skate, person):
    sx1, sy1, sx2, sy2 = skate
    px1, py1, px2, py2 = person
    skate_center_x = (sx1 + sx2) / 2
    skate_center_y = (sy1 + sy2) / 2

    # Скейт внутри границ человека по горизонтали
    in_width = px1 < skate_center_x < px2
    # Скейт в районе ног (нижняя четверть тела)
    person_height = py2 - py1
    in_height = (py2 - person_height * 0.25) < skate_center_y < (py2 + person_height * 0.15)
    return in_width and in_height


def save_to_history(entry):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            history = []
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    raw_path = os.path.join(RESULT_DIR, "temp_raw.mp4")
    web_filename = f"web_{datetime.now().strftime('%H%M%S')}_{file.filename.split('.')[0]}.mp4"
    web_path = os.path.join(RESULT_DIR, web_filename)

    cap = cv2.VideoCapture(temp_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    violation_detected = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model.track(frame, persist=True, verbose=False, device=DEVICE, conf=0.45, imgsz=640)

        persons = []
        skateboards = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                coords = box.xyxy[0].cpu().numpy()
                if cls == 0:
                    persons.append(coords)  # Человек
                elif cls == 36:
                    skateboards.append(coords)  # Скейтборд

        used_person_indices = set()
        for s_box in skateboards:
            riding = False
            for i, p_box in enumerate(persons):
                if i not in used_person_indices and is_riding(s_box, p_box):
                    riding = True
                    used_person_indices.add(i)
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

    # FFmpeg конвертация для воспроизведения в браузере
    subprocess.run([
        'ffmpeg', '-y', '-i', raw_path,
        '-vcodec', 'libx264', '-preset', 'ultrafast', '-crf', '28',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', web_path
    ], capture_output=True)

    if os.path.exists(raw_path): os.remove(raw_path)
    if os.path.exists(temp_path): os.remove(temp_path)

    # Сохранение в историю
    history_entry = {
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "filename": file.filename,
        "violation": "Да" if violation_detected else "Нет",
        "frames": frame_count
    }
    save_to_history(history_entry)

    return {
        "violation": "Yes" if violation_detected else "No",
        "video_url": f"/static/results/{web_filename}"
    }


@app.get("/get_history")
async def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/export_excel")
async def export_excel():
    if not os.path.exists(HISTORY_FILE):
        return {"error": "История пуста"}

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df.columns = ['Дата и время', 'Название файла', 'Нарушение', 'Кол-во кадров']

    excel_path = os.path.join(RESULT_DIR, "skate_report.xlsx")
    df.to_excel(excel_path, index=False)
    return FileResponse(excel_path, filename="Report_Skate_Monitoring.xlsx")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)