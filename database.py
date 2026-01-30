import json
import os
import pandas as pd

HISTORY_FILE = "history.json"

def save_to_history(entry):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except: history = []
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def get_all_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def create_excel_report(output_path):
    data = get_all_history()
    if not data: return False
    df = pd.DataFrame(data)
    df.columns = ['Дата и время', 'Название файла', 'Нарушение', 'Кол-во кадров']
    df.to_excel(output_path, index=False)
    return True