from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from create_sample_set import get_train_samples

model_path = 'models/'

model = load_model(model_path + 'model.h5')

test_samples, test_labels = get_train_samples(200)
predictions = model.predict(test_samples, batch_size=10, verbose=0)
classified_predictions = np.argmax(predictions, axis=-1)

cm = confusion_matrix(test_labels, classified_predictions)
plt.figure()
plt.title('Confusion Matrix')
sn.heatmap(cm, cmap='coolwarm', annot=True, fmt='.5g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
