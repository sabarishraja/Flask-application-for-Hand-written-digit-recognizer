import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tf_keras.models import load_model
from tf_keras.utils import to_categorical
# Load the trained model
model = load_model('mnist_model.h5')

# Load the MNIST test data
from tf_keras.datasets import mnist
(_, _), (X_test, y_test) = mnist.load_data()

# Preprocess the test data
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_test = to_categorical(y_test, num_classes=10)

y_test = np.argmax(y_test, axis=1)

# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Visualize misclassified images
misclassified_indices = np.where(y_pred_classes != y_test)[0]

for i in range(5):  # Show 5 misclassified examples
    idx = misclassified_indices[i]
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    plt.title(f"True: {y_test[idx]}, Predicted: {y_pred_classes[idx]}")
    plt.show()