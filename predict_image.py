import numpy as np
import torch
from ultralytics import YOLO
import numpy as np
import torch
import os
from pathlib import Path

# Charger le modèle YOLOv8
model = YOLO('./runs/classify/train/weights/last.pt')

# Faire une prédiction sur une nouvelle image
results = model('/content/gdrive/MyDrive/image-cassaca-classification-yolov8/CBSD4.jpg')

# Obtenir les noms des classes
names_dict = results[0].names

probs = results[0].probs.numpy()

max_index = probs.top1

predicted_class = names_dict[max_index]

print(f"\nL'image appartient à la classe : {predicted_class}")