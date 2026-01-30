import torch
from ultralytics import YOLO

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolo11m.pt').to(DEVICE)


def is_riding(skate, person):
    sx1, sy1, sx2, sy2 = skate
    px1, py1, px2, py2 = person
    skate_center_x = (sx1 + sx2) / 2
    skate_center_y = (sy1 + sy2) / 2

    in_width = px1 < skate_center_x < px2
    person_height = py2 - py1
    in_height = (py2 - person_height * 0.25) < skate_center_y < (py2 + person_height * 0.15)
    return in_width and in_height