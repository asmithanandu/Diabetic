import base64
import numpy as np
from io import BytesIO
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from PIL import Image

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ===========================
# Load your model globally
# ===========================
# Make sure 'efficient_model.h5' is in your project root or provide full path
efficient_model = load_model('DiabeticApp/models/efficient_model.h5')
# This ensures the graph/session is maintained globally
graph = tf.compat.v1.get_default_graph()

# ===========================
# Render predict page
# ===========================
def Predict(request):
    return render(request, 'predict.html', {'data': '', 'img': None})

# ===========================
# Handle image upload & prediction
# ===========================
@csrf_exempt
def PredictAction(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get('t1')
        if not uploaded_file:
            return render(request, 'predict.html', {'data': 'No file uploaded!', 'img': None})

        # Open image and preprocess
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((224, 224))  # Resize according to your model input
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize

        # Predict using global model & graph
        global graph
        with graph.as_default():
            preds = efficient_model.predict(img_array)
            predicted_class = np.argmax(preds, axis=1)[0]

        # Map class index to meaningful label
        class_labels = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        result = class_labels.get(predicted_class, 'Unknown')

        # Convert uploaded image to base64 to display in template
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return render(request, 'predict.html', {'data': f'Prediction: {result}', 'img': img_str})

    return render(request, 'predict.html', {'data': '', 'img': None})
