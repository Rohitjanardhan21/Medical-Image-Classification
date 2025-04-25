import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
LABEL_MAP = {
    "COVID-CT": 0,
    "COVID-XRay": 1,
    "Pneumonia-Bacterial": 2,
    "Pneumonia-Viral": 3,
    "Tuberculosis": 4
}


def load_and_preprocess_single_image(img_path):
    """Loads and preprocesses a single image.

    Args:
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image, or None if an error occurs.
    """
    try:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure correct color format
            img = img.astype(np.float32) / 255.0  # Normalize to float32
            return img
        return None
    except Exception as e:
        print(f"Error loading image: {img_path}, error: {e}")
        return None


def preprocess_data(test_size=0.2, random_state=42, stratify=True):
    """Loads and preprocesses the dataset.

    Args:
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.
        stratify (bool, optional): Whether to stratify the data.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
            X_train (np.ndarray): Training images.
            X_test (np.ndarray): Testing images.
            y_train (np.ndarray): Training labels.
            y_test (np.ndarray): Testing labels.
    """
    data = []
    labels = []
    datasets = [
        (r"C:\Users\rohit\OneDrive\Desktop\DaTaSeT 2\covid_ct\Disease", "COVID-CT"),
        (r"C:\Users\rohit\OneDrive\Desktop\DaTaSeT 2\covid_xray\Disease", "COVID-XRay"),
        (r"C:\Users\rohit\OneDrive\Desktop\DaTaSeT 2\pneumonia_bacterial", "Pneumonia-Bacterial"),
        (r"C:\Users\rohit\OneDrive\Desktop\DaTaSeT 2\pneumonia_viral", "Pneumonia-Viral"),
        (r"C:\Users\rohit\OneDrive\Desktop\DaTaSeT 2\tuberculosis", "Tuberculosis")
        # Removed the normal path
    ]

    for path, label_key in datasets:
        for filename in os.listdir(path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(path, filename)
                data.append((img_path, LABEL_MAP[label_key]))

    np.random.shuffle(data)  # Shuffle the data

    all_images = []
    all_labels = []
    for img_path, label in data:
        img = load_and_preprocess_single_image(img_path)
        if img is not None:
            all_images.append(img)
            all_labels.append(label)

    X = np.array(all_images)
    y = to_categorical(all_labels, num_classes=len(LABEL_MAP))  # Use length of the updated label map
    if stratify:
        y_stratify = np.array(all_labels)
    else:
        y_stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y_stratify)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_data()
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("X_train dtype:", X_train.dtype)
    print("y_train dtype:", y_train.dtype)