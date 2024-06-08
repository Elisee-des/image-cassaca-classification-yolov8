from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch

app = Flask(__name__)

# Charger le modèle YOLOv8
model = YOLO('./runs/classify/train6/weights/last.pt')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_base64 = data['image']
        
        # Retirer le préfixe "data:image/jpeg;base64,"
        if image_base64.startswith('data:image'):
            image_base64 = image_base64.split(",")[1]
        
        # Décoder l'image base64
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Convertir l'image en tableau numpy
        image_np = np.array(image)
        
        # Faire une prédiction sur l'image
        results = model(image_np)
        
        # Obtenir les noms des classes
        names_dict = results[0].names
        
        # Obtenir les probabilités
        probs = results[0].probs.numpy()
        
        # Trouver l'index de la classe avec la probabilité maximale
        max_index = probs.argmax()
        
        # Obtenir le nom de la classe prédite
        predicted_class = names_dict[max_index]
        
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
