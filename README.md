# Medical Image Classification Project

## Overview

This project is a medical image classification system that uses deep learning to detect various diseases from medical images such as X-rays, MRI scans, and CT scans.  It aims to assist healthcare professionals in diagnosing conditions like tuberculosis, COVID-19, and pneumonia by providing a diagnosis and suggesting potential next steps.

## Features

* **Disease Detection:** Identifies diseases such as tuberculosis, COVID-19, and pneumonia.
* **Image Source Flexibility:** Accepts input from various medical imaging sources, including X-rays, MRI scans, and CT scans.
* **Diagnosis Output:** Provides a diagnosis of the detected condition.
* **Recommendation of Next Steps:** Suggests potential next steps or actions based on the diagnosis.
* **Web Interface:** User-friendly web interface for uploading images and viewing results.

## Project Structure and File Description

The project consists of the following key files:

* **`app.py`**: This is the main Flask application file. It handles the web interface, image uploads, and integrates the model for making predictions.
    * Sets up the Flask web application.
    * Defines routes for handling image uploads and displaying results.
    * Calls the prediction module (`prediction.py`) to classify uploaded images.
    * Renders the `index.html` template.
* **`model.py`**: While the provided code doesn't have a separate `model.py`, in a typical project, this file would contain the definition of the deep learning model architecture. However, the model architecture is defined inside `train_model.py`.
    * **Model Definition**: Defines the CNN architecture (e.g., using TensorFlow/Keras). The current code uses a Sequential model with Conv2D, MaxPooling2D, and Dense layers. It could also include loading a pre-trained model (transfer learning).
    * **Compilation**: Specifies the optimizer, loss function, and metrics for training the model.
* **`prediction.py`**: This file contains the logic for loading the trained model and making predictions on new images.
    * Loads the trained model from an h5 file.
    * Preprocesses input images to match the format expected by the model.
    * Uses the loaded model to predict the disease from the image.
    * Returns the diagnosis and confidence of the prediction.
* **`train_model.py`**: This script is used to train the deep learning model.
    * Loads and preprocesses the training data.
    * Defines the model architecture (if not in a separate `model.py`).
    * Trains the model using the training data and a specified number of epochs.
    * Saves the trained model to a file.
* **`index.html`**: This is the HTML template for the web interface.
    * Provides an interface for users to upload medical images.
    * Displays the uploaded image, the diagnosis, confidence score, and recommended next steps.
    * Uses JavaScript to handle user interactions and display results.

## How to Run the Application

Follow these steps to run the medical image classification application:

1.  **Clone the Repository:**

    ```bash
    git clone <your_repository_url>
    cd medical-image-classification
    ```

2.  **Install Dependencies:** It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    pip install -r requirements.txt # Install the required packages
    ```

    *(Note: You might need to create a `requirements.txt` file listing the dependencies, including TensorFlow, Flask, OpenCV, etc. A sample `requirements.txt` is shown below.)*

3.  **Prepare the Dataset:**

    * Download or obtain your medical image dataset. The dataset should be organized into directories, with each directory representing a different disease class (e.g., `COVID-CT`, `COVID-XRay`, `Pneumonia-Bacterial`, `Pneumonia-Viral`, and `Normal`).
    * Modify the `DATA_DIR` variable in `train_model.py` to point to the location of your dataset.
    * Ensure that the data loading and preprocessing steps in `train_model.py` and `prediction.py` are consistent with your dataset structure.

4.  **Train the Model (Optional):**

    * If you want to use a pre-trained model, you can skip this step, but make sure you have the trained model file (e.g., `medical_image_classifier.h5`).
    * If you need to train the model, run the `train_model.py` script:

        ```bash
        python train_model.py
        ```

        * This will train the model and save it to `medical_image_classifier.h5` (by default).
        * You can adjust the training parameters (e.g., epochs, batch size) in the `train_model.py` script.

5.  **Run the Flask Application:**

    * Run the `app.py` script to start the web application:

        ```bash
        python app.py
        ```

    * The application will start, and you can access it in your web browser (typically at `http://127.0.0.1:5000/`).

6.  **Use the Application:**

    * Open the web application in your browser.
    * Upload a medical image (X-ray, MRI, or CT scan) using the "Choose File" button.
    * Click the "Classify Image" button to upload and process the image.
    * The application will display the uploaded image, the diagnosis, confidence score, and recommended next steps.

##  `requirements.txt`

```text
tensorflow==2.15.0 # Or your compatible TensorFlow version
Flask==2.3.2
opencv-python==4.9.0.54
scikit-learn==1.2.2
Werkzeug==3.0.1
Model
