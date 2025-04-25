import sys
import os

# Get the path to the parent directory (project root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the Python path
sys.path.append(project_root)

#  No need to import the whole module, import directly.
from preprocessing import preprocess_data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def plot_disease_probabilities(model, X_sample, class_names):
    # Get prediction probabilities for a sample image
    predictions = model.predict(X_sample[np.newaxis, ...])

    # Create a horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(class_names, predictions[0])
    plt.xlabel('Probability')
    plt.ylabel('Disease Classes')
    plt.title('Disease Probability Distribution')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()


def train_model():
    # Load and preprocess data
    try:
        X_train, X_test, y_train, y_test = preprocess_data()
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return  # Exit if preprocessing fails

    # Define class names (replace with your actual class names)
    class_names = ['COVID-CT', 'COVID-XRay', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'Tuberculosis']

    # Build and compile model
    model = build_model(input_shape=X_train.shape[1:], num_classes=len(class_names)) # Use len(class_names)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    try:
        history = model.fit(X_train, y_train,
                            epochs=10,
                            validation_data=(X_test, y_test),
                            batch_size=32)
    except Exception as e:
        print(f"Error during model training: {e}")
        return  # Exit if training fails

    # Evaluate model on the entire test dataset
    try:
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"\n✅ Overall Test Accuracy: {test_accuracy*100:.2f}%")
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return  # Exit if evaluation fails

    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot disease probabilities for a sample image
    if len(X_test) > 0:
        sample_idx = 0  # Choose an example from test set
        sample_image = X_test[sample_idx]
        plot_disease_probabilities(model, sample_image, class_names)

    # Save the model
    try:
        model.save("medical_image_classifier.h5")
        print("✅ Model trained and saved as medical_image_classifier.h5")
    except Exception as e:
        print(f"Error during saving the model: {e}")
        return  # Exit if saving model fails


if __name__ == "__main__":
    train_model()