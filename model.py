from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from preprocessing import preprocess_data
import matplotlib.pyplot as plt
import numpy as np

LABEL_MAP = {
    0: "COVID-CT",
    1: "COVID-XRay",
    2: "Pneumonia-Bacterial",
    3: "Pneumonia-Viral",
    4: "Tuberculosis"
}

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def plot_class_probabilities(probabilities):
    labels = list(LABEL_MAP.values())
    plt.figure(figsize=(8, 5))
    plt.bar(labels, probabilities, color='skyblue')
    plt.ylim([0, 1])
    plt.title("Prediction Probabilities")
    plt.ylabel("Probability")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

def train_model():
    X_train, X_test, y_train, y_test = preprocess_data()

    # ✅ Changed num_classes to 5
    model = build_model(input_shape=X_train.shape[1:], num_classes=len(LABEL_MAP))
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"✅ Final test accuracy: {accuracy * 100:.2f}%")

    model.save("medical_model.h5")
    print("✅ Model trained and saved as medical_model.h5")

    random_index = np.random.randint(0, len(X_test))
    sample_image = np.expand_dims(X_test[random_index], axis=0)
    predictions = model.predict(sample_image)[0]

    print("🧪 Sample prediction probabilities:")
    for i, prob in enumerate(predictions):
        print(f"  {LABEL_MAP[i]}: {prob*100:.2f}%")

    plot_class_probabilities(predictions)

if __name__ == "__main__":
    train_model()