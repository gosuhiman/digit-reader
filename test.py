from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os

# Paths
model_path = 'models'
data_path = 'data'
test_data_path = os.path.join(data_path, 'my-digits')
# test_data_path = os.path.join(data_path, 'mnist', 'images', 'test')
batch_size = 1000

image_size = (28, 28)
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

test_batches = ImageDataGenerator().flow_from_directory(
    test_data_path,
    target_size=image_size,
    classes=classes,
    class_mode='sparse',
    batch_size=batch_size,
    color_mode='grayscale',
    shuffle=False
)

model = load_model(os.path.join(model_path, 'model.h5'))

model.evaluate(test_batches)

predictions = model.predict(test_batches)
classified_predictions = np.argmax(predictions, axis=-1)

cm = confusion_matrix(test_batches.labels, classified_predictions)
plt.figure()
plt.title('Confusion Matrix')
sn.heatmap(cm, cmap='coolwarm', annot=True, fmt='.5g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
