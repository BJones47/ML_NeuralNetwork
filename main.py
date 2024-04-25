import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt

# Function to plot learning curves
def plot_learning_curves(history):
    plt.figure(figsize=(12, 6))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# model for nn dropout a little high for overfitting
def build_model():
    model = Sequential([
        Dense(512, activation='relu', input_shape=(784,)),
        Dropout(0.5),  
        Dense(256, activation='relu'),
        Dropout(0.5),  
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the dataset from a CSV file
df = pd.read_csv('MNIST_HW4.csv')

# Assuming the first column 'label' contains the labels
y = df['label'].values
X = df.drop('label', axis=1).values  

# Normalize the pixel values from 0-255 to 0-1 range
X = X.astype('float32') / 255

# Convert class vectors to binary class matrices
y = to_categorical(y, 10)

# Prepare cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
scores = []
results_summary = []

# used for the learning curve plot
histories = []
# Convert categorical matrix back to labels for stratified kfold
targets = np.argmax(y, axis=1)

for train, test in kfold.split(X, targets):
    model = build_model()
    # Fit data to model
    history = model.fit(X[train], y[train],
                        epochs=25,
                        batch_size=32,
                        verbose=1,
                        validation_data=(X[test], y[test]),
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])  

    # Evaluate the model
    score = model.evaluate(X[test], y[test], verbose=0)
    scores.append(score)
    histories.append(history)
    result_string = f'Fold {fold_no}: Loss = {score[0]:.4f}, Accuracy = {score[1] * 100:.2f}%'
    results_summary.append(result_string)
    fold_no += 1

# Average scores
for result in results_summary:
    print(result)

avg_loss = np.mean([score[0] for score in scores])
avg_accuracy = np.mean([score[1] for score in scores])
print(f'Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy * 100:.2f}%')

for i, history in enumerate(histories):
    plot_learning_curves(history)
