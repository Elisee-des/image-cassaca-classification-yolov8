# Importer le module YOLO
from ultralytics import YOLO
DATA_DIR = 'dataset'

# Charger le modèle YOLOv8 pour la classification
model = YOLO('yolov8n-cls.pt')

# Entraîner le modèle
model.train(data=DATA_DIR, epochs=2, imgsz=224)

# Évaluer le modèle sur l'ensemble de validation
metrics = model.val()

# Afficher les métriques d'évaluation
print(metrics)

# Enregistrer le modèle entraîné
model_path = 'yolov8n-cls-trained.pt'
model.save(model_path)
