import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Store the paths to the metadata and image folders on my computer
metadata_path = "/Users/vancityaziz/Desktop/ham10000-svm/Ham10000imgs/HAM10000_metadata.csv"
image_folder_1 = "/Users/vancityaziz/Desktop/ham10000-svm/Ham10000imgs/HAM10000_images_part_1"
image_folder_2 = "/Users/vancityaziz/Desktop/ham10000-svm/Ham10000imgs/HAM10000_images_part_2"

# Use Pandas to create a Pandas DataFrame from the metadata CSV file
print("Loading metadata...") # Indicates where the code is in the process so we don't thnk it's stuck
df = pd.read_csv(metadata_path)

# Define a function to load image, check if it exists in either folder, and return the path
def load_image(img_id):
    print("Loading images...")
    filename = img_id + '.jpg'
    path1 = os.path.join(image_folder_1, filename)
    path2 = os.path.join(image_folder_2, filename)
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        return None

df['img_path'] = df['image_id'].apply(load_image)
df = df[df['img_path'].notnull()]  # Filter out rows where image path is None

# Resize and extract features
print("Extracting image features...")
def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))  # Resize to 64x64
    return img.flatten()

X = []
for path in tqdm(df['img_path']):
    X.append(extract_features(path))
X = np.array(X)

# Label encoding the target variable
le = LabelEncoder()
y = le.fit_transform(df['dx'])

# Train/Val/Test split 70/15/15 
# First take out 15% for test, then split remaining 85% into train and val (70/15 overall) 
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)


# Scale 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train SVM 
print("Training SVM...")
clf = SVC(kernel='linear')  # Using linear kernel for simplicity
clf.fit(X_train, y_train)

# Evaluate 
print("\nValidation Performance:")
y_val_pred = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

print("\nTest Performance:")
y_test_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, target_names=le.classes_))
