import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, TOP, BOTTOM, messagebox
from PIL import Image, ImageTk
import threading  # Import the threading module

# Constants
MODEL_PATH = 'models/medical_image_classifier.h5'
IMAGE_SIZE = (224, 224)
DISPLAY_SIZE = (400, 400)  # Size for display in the GUI
LABEL_MAP = {
    0: "COVID-CT",
    1: "COVID-XRay",
    2: "Pneumonia-Bacterial",
    3: "Pneumonia-Viral",
    4: "Tuberculosis"
}

# Load model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded for Flask app.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def preprocess_image(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image_from_path(image_path):
    if model is None:
        return "Model not loaded", 0, None  # Return None for image

    img = cv2.imread(image_path)
    if img is None:
        return "Invalid image", 0, None  # Return None for image

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    diagnosis = LABEL_MAP.get(predicted_class, "Unknown") # Handle potential out-of-range if model isn't retrained
    return diagnosis, confidence, img  # Return the image

def select_image_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy() #destroy the root
    return file_path

def display_results(image_path):
    diagnosis, confidence, original_img = predict_image_from_path(image_path) #unpack 3 values

    # Create a new window to display results
    result_window = tk.Toplevel()
    result_window.title("Medical Image Classification Results")
    result_window.geometry("500x600")

    # Create frames
    image_frame = Frame(result_window)
    image_frame.pack(side=TOP, pady=10)

    result_frame = Frame(result_window)
    result_frame.pack(side=TOP, pady=10)

    button_frame = Frame(result_window)
    button_frame.pack(side=BOTTOM, pady=20)

    # Display the image
    if original_img is not None: #check if image is valid
        display_img = cv2.resize(original_img, DISPLAY_SIZE)
        img = Image.fromarray(display_img)
        img_tk = ImageTk.PhotoImage(image=img)

        img_label = Label(image_frame, image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection
        img_label.pack()
    else:
        Label(image_frame, text="No image to display").pack()

    # Display the diagnosis result
    result_text = f"Diagnosis: {diagnosis}\nConfidence: {confidence * 100:.2f}%"
    result_label = Label(result_frame, text=result_text, font=("Arial", 16), pady=10)
    result_label.pack()

    # Risk level indicator based on confidence
    risk_text = ""
    risk_color = ""
    if confidence > 0.85:
        risk_text = "High confidence in diagnosis"
        risk_color = "#4CAF50"  # Green
    elif confidence > 0.60:
        risk_text = "Moderate confidence in diagnosis"
        risk_color = "#FFC107"  # Yellow/Amber
    else:
        risk_text = "Low confidence in diagnosis, recommend further testing"
        risk_color = "#F44336"  # Red

    risk_label = Label(result_frame, text=risk_text, font=("Arial", 14), fg=risk_color, pady=5)
    risk_label.pack()

    # Description of the condition
    descriptions = {
        "COVID-CT": "COVID-19 detected in CT scan. Shows characteristic ground-glass opacities in lungs.",
        "COVID-XRay": "COVID-19 detected in X-Ray. Shows bilateral infiltrates often in lower zones.",
        "Pneumonia-Bacterial": "Bacterial pneumonia shows as dense consolidation often in one lobe.",
        "Pneumonia-Viral": "Viral pneumonia typically presents with diffuse interstitial patterns.",
        "Tuberculosis": "Tuberculosis often shows upper lobe infiltrates and cavitation.",
    }

    description_label = Label(result_frame, text=f"Description:\n{descriptions.get(diagnosis, '')}",
                                     font=("Arial", 12), wraplength=400, justify="left", pady=10)
    description_label.pack()

    # Next steps button
    def show_next_steps():
        next_steps_window = tk.Toplevel()
        next_steps_window.title("Recommended Next Steps")
        next_steps_window.geometry("400x300")

        recommendations = {
            "COVID-CT": "Isolate patient, monitor oxygen saturation, consider antiviral therapy if eligible.",
            "COVID-XRay": "Isolate patient, monitor vital signs, provide supportive care.",
            "Pneumonia-Bacterial": "Prescribe appropriate antibiotics, supportive care, follow-up imaging in 4-6 weeks.",
            "Pneumonia-Viral": "Supportive care, monitor for secondary bacterial infection, rest and hydration.",
            "Tuberculosis": "Isolation, full TB workup, initiate anti-TB therapy as appropriate.",
        }

        Label(next_steps_window, text="Recommended Next Steps", font=("Arial", 14, "bold"), pady=10).pack()
        Label(next_steps_window, text=recommendations.get(diagnosis, ""),
              font=("Arial", 12), wraplength=350, justify="left", pady=10).pack()

        Label(next_steps_window, text="Note: This is a decision support tool only.\nPlease follow clinical protocols and use professional judgment.",
              font=("Arial", 10, "italic"), pady=10).pack()

        Button(next_steps_window, text="Close", command=next_steps_window.destroy,
               font=("Arial", 12), padx=20, pady=5).pack(pady=10)

    next_steps_button = Button(button_frame, text="Show Recommendations", command=show_next_steps,
                                     font=("Arial", 12), padx=10, pady=5)
    next_steps_button.pack(side=tk.LEFT, padx=10)

    print(f"\n🩺 Diagnosis: {diagnosis} (Confidence: {confidence * 100:.2f}%)\n")

    # Focus the result window
    result_window.focus_force()

    # Close button
    close_button = Button(button_frame, text="Close", command=result_window.destroy,
                             font=("Arial", 12), padx=20, pady=5)
    close_button.pack()

def upload_action():
    image_path = select_image_file()
    if image_path:
        # Create a thread to handle the image processing and display
        thread = threading.Thread(target=display_results, args=(image_path,))
        thread.start()  # Start the thread
    else:
        print("❗ No file selected.")
        messagebox.showinfo("No File Selected", "Please select an image to upload.")

# CLI menu
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Medical Image Classifier")
    root.geometry("400x300")

    # Header
    header_label = Label(root, text="Medical Image Classification", font=("Arial", 16, "bold"), pady=20)
    header_label.pack()

    # Instructions
    instruction_label = Label(root, text="Upload a medical image for classification", font=("Arial", 12), pady=10)
    instruction_label.pack()

    # Upload button
    upload_button = Button(root, text="Upload Image", command=upload_action, font=("Arial", 12), padx=20, pady=10)
    upload_button.pack(pady=20)

    # Exit button
    exit_button = Button(root, text="Exit", command=root.destroy, font=("Arial", 12), padx=20, pady=10)
    exit_button.pack(pady=10)

    root.mainloop()