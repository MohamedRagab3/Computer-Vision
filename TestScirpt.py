import os
import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Function to preprocess the image for testing
def preprocess_image(image, scaler):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    resized_image = cv2.resize(image_gray, (224, 224))
    
    flattened_image = resized_image.flatten()
    
    # Use the same scaler from training
    normalized_image = scaler.transform([flattened_image])  
    
    return normalized_image

# Load the trained SVM model and components
print("Loading the trained SVM model and other components...")
model_path = r"C:\\Users\\hp\\Desktop\\CV Project\\SavedModels\\svm_model.pkl"
scaler_path = r"C:\\Users\\hp\\Desktop\\CV Project\\SavedModels\\scaler.pkl"
label_encoder_path = r"C:\\Users\\hp\\Desktop\\CV Project\\SavedModels\\label_encoder.pkl"
# Path to the test data directory
DATA_DIR = r"C:\\Users\\hp\\Desktop\\CV Project\\Test Data\\Test samples for classification"

svm_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

print("Model, scaler, and label encoder loaded successfully!")

# Variables for storing results
y_true = []
y_pred = []

# Processing the test data
print("\nProcessing test data...")
for folder_name in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder_name)
    
    if os.path.isdir(folder_path):
        prefix, _ = folder_name.split("___", 1) 
        
        for img_filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_filename)
            img_data = cv2.imread(img_path)

            if img_data is not None:
                preprocessed_img = preprocess_image(img_data, scaler)
                
                prediction = svm_model.predict(preprocessed_img)
                
                y_true.append(prefix)  # Append the prefix as the true label
                y_pred.append(prediction[0])  # Append the encoded predicted label

                # Optional: Print each prediction
                print(f"Image: {img_filename}, True Label: {prefix}, Predicted Label: {label_encoder.inverse_transform([prediction[0]])[0]}")

# Encode both true and predicted labels using the label encoder
y_true_encoded = label_encoder.transform(y_true)
y_pred_encoded = np.array(y_pred)

# Calculate total accuracy
accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
print(f"\nTotal Accuracy: {accuracy * 100:.2f}%")

# Display confusion matrix
conf_matrix = confusion_matrix(y_true_encoded, y_pred_encoded, labels=range(len(label_encoder.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap="viridis")
plt.title("Confusion Matrix")
plt.show()