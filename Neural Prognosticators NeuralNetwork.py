import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Example file paths (replace with your actual EEG data file paths)
file_paths = [
    "/work/sjajee/stage1_labeled/0_0.csv",
    "/work/sjajee/stage1_labeled/0_1.csv",
    "/work/sjajee/stage1_labeled/1_0.csv",
    "/work/sjajee/stage1_labeled/1_1.csv",
    "/work/sjajee/stage1_labeled/2_0.csv",
    "/work/sjajee/stage1_labeled/2_1.csv",
    "/work/sjajee/stage1_labeled/3_0.csv",
    "/work/sjajee/stage1_labeled/3_1.csv",
    "/work/sjajee/stage1_labeled/4_0.csv",
    "/work/sjajee/stage1_labeled/4_1.csv",
    "/work/sjajee/stage1_labeled/5_0.csv",
    "/work/sjajee/stage1_labeled/5_1.csv",
    "/work/sjajee/stage1_labeled/6_0.csv",
    "/work/sjajee/stage1_labeled/6_1.csv",
    "/work/sjajee/stage1_labeled/7_0.csv",
    "/work/sjajee/stage1_labeled/7_1.csv",
]

# Load and preprocess your data
combined_data = pd.concat([pd.read_csv(file) for file in file_paths])
X = combined_data.iloc[:, :-1].values  # Features (assuming the last column is the label)
y = combined_data.iloc[:, -1].values   # Labels

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Neural network architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(test_loss, test_acc))

# Predict on test data
y_test_pred_probs = model.predict(X_test)
y_test_pred = np.argmax(y_test_pred_probs, axis=1)

# Calculate precision, recall, and F1 score for the test set
precision_test = precision_score(y_test, y_test_pred, average='weighted')
recall_test = recall_score(y_test, y_test_pred, average='weighted')
f1_test = f1_score(y_test, y_test_pred, average='weighted')

print("Test Precision: {:.4f}, Test Recall: {:.4f}, Test F1 Score: {:.4f}".format(precision_test, recall_test, f1_test))
