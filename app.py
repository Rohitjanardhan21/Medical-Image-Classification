from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'  # Store uploaded images here
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}  # Allowed file types

# Load the trained model
model = load_model('models/medical_image_classifier.h5')


class_names = ['COVID-CT', 'COVID-XRay', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'Tuberculosis']


DESCRIPTIONS = {
    "COVID-CT": "COVID-19 detected in CT scan. Shows characteristic ground-glass opacities in lungs.",
    "COVID-XRay": "COVID-19 detected in X-Ray. Shows bilateral infiltrates often in lower zones.",
    "Pneumonia-Bacterial": "Bacterial pneumonia shows as dense consolidation often in one lobe.",
    "Pneumonia-Viral": "Viral pneumonia typically presents with diffuse interstitial patterns.",
    "Tuberculosis": "Tuberculosis often shows upper lobe infiltrates and cavitation."
}


RECOMMENDATIONS = {
    "COVID-CT": "Isolate patient, monitor oxygen saturation, consider antiviral therapy if eligible.",
    "COVID-XRay": "Isolate patient, monitor vital signs, provide supportive care.",
    "Pneumonia-Bacterial": "Prescribe appropriate antibiotics, supportive care, follow-up imaging in 4-6 weeks.",
    "Pneumonia-Viral": "Supportive care, monitor for secondary bacterial infection, rest and hydration.",
    "Tuberculosis": "Isolation, full TB workup, initiate anti-TB therapy as appropriate."
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_probability_chart(prediction_array):
    plt.figure(figsize=(10, 6))
    plt.barh(class_names, prediction_array[0])
    plt.xlabel('Probability')
    plt.ylabel('Disease Classes')
    plt.title('Disease Probability Distribution')
    plt.xlim(0, 1)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def predict_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index] * 100
    predicted_class = class_names[index]
    probability_chart = generate_probability_chart(prediction)
    all_probabilities = {class_name: float(prob) * 100 for class_name, prob in zip(class_names, prediction[0])}

    return predicted_class, confidence, probability_chart, all_probabilities

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            diagnosis, confidence, probability_chart, all_probabilities = predict_image(file_path)
            confidence = round(confidence, 2)
            description = DESCRIPTIONS.get(diagnosis, "No description available.")
            recommendation = RECOMMENDATIONS.get(diagnosis, "No specific recommendations available.")

            return render_template(
                'index.html',
                image_path=file_path,
                prediction=diagnosis,
                confidence=confidence,
                show_results=True,
                description=description,
                recommendation=recommendation,
                probability_chart=probability_chart,
                all_probabilities=all_probabilities
            )
        else:
            return render_template('index.html', error='Invalid file type. Allowed types are: png, jpg, jpeg, gif, bmp')
    return render_template('index.html', show_results=False)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)