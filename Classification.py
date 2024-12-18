import os
import cv2
import time
import joblib
import numpy as np
from tqdm import tqdm
from random import shuffle
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# There is still one important step is: Feature Extraction using (SIFT)
# Preprocessing
def preprocess_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)  
    return img

# Create training data with labels
def create_data(DATA_DIR, IMG_SIZE):
    data = []
    label_mapping = {}
    current_label = 0

    # Mapping labels to indices
    for folder_name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if os.path.isdir(folder_path):
            prefix, label = folder_name.split("___")  # Split "Apple___Apple_scab" -> "Apple", "Apple_scab"

            # If prefix not in label_mapping, assign new label
            if prefix not in label_mapping:
                label_mapping[prefix] = current_label  # Assign dynamic label
                current_label += 1

            # Process images in the folder
            for img_filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_filename)
                img_data = cv2.imread(img_path, 0)  # Load as grayscale
                if img_data is not None:
                    resized_img = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
                    preprocessed_img = preprocess_image(resized_img)
                    data.append([preprocessed_img, label_mapping[prefix]])

    # Shuffle the data
    shuffle(data)
    return data, label_mapping

# Reshape and normalize data
def reshape_and_normalize(data):
    x = np.array([i[0] for i in data])  # Features (images)
    y = np.array([i[1] for i in data])  # Labels
    return x, y

# Paths and parameters
TRAIN_DIR = r"C:\\Users\\hp\\Desktop\\CV Project\\Project Data\\Train\\images"
TEST_DIR = r"C:\\Users\\hp\\Desktop\\CV Project\\Project Data\\Val\\images"
IMG_SIZE = 224  # Image size for resizing

# Create and preprocess training data
print("Creating Training Data...")
train_data, train_label_mapping = create_data(TRAIN_DIR, IMG_SIZE)
print("Creating Testing Data....")
test_data, test_label_mapping = create_data(TEST_DIR, IMG_SIZE)

print("Number of classes:", len(train_label_mapping))

print("Class Label Mapping (Class Name -> Encoded Label):")
for label, encoding in train_label_mapping.items():
    print(f"{label} -> {encoding}")

# Reshape and normalize data
x_train, y_train = reshape_and_normalize(train_data)
x_test, y_test = reshape_and_normalize(test_data)

# Reshape images to 2D for SVM    # Flatten images
x_train_flat = x_train.reshape(len(x_train), -1)  
x_test_flat = x_test.reshape(len(x_test), -1)

# Scale the data (SVM requires scaled data)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

# Encode labels
# Create and fit LabelEncoder based on the class names from label_mapping
label_encoder = LabelEncoder()

# Set the classes_ attribute of LabelEncoder based on label_mapping
label_encoder.classes_ = np.array(list(train_label_mapping.keys()))

# Encode labels using LabelEncoder
y_train_encoded = label_encoder.transform([list(train_label_mapping.keys())[i] for i in y_train])
y_test_encoded = label_encoder.transform([list(train_label_mapping.keys())[i] for i in y_test])

print("Classes in the LabelEncoder:", label_encoder.classes_)

# Train the SVM model
print("\nTraining the SVM model...")
start_train_time = time.time()
svm_model = SVC(kernel='rbf', C=1)
svm_model.fit(x_train_scaled, y_train_encoded)
end_train_time = time.time()

# Predict using the trained model
print("Evaluating the SVM model...")
start_eval_time = time.time()
y_pred = svm_model.predict(x_test_scaled)
end_eval_time = time.time()

# Calculate accuracy
test_accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\nTest Accuracy: {test_accuracy*100:.1f} %")


# Print training and evaluation time
print(f"\nTraining Time: {end_train_time - start_train_time:.2f} seconds")
print(f"Evaluation Time: {end_eval_time - start_eval_time:.2f} seconds")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred)
print("\nConfusion Matrix:")

# Display Confusion Matrix with better visuals
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()



# Save the trained SVM model &  Save the label encoder &  Save the scaler
joblib.dump(svm_model, "C:\\Users\\hp\\Desktop\\svm_model.pkl")

joblib.dump(label_encoder, "C:\\Users\\hp\\Desktop\\label_encoder.pkl")

joblib.dump(scaler, "C:\\Users\\hp\\Desktop\\scaler.pkl")  

print("Model saved successfully to Desktop!")