from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
model_path = 'models/'
data_path = 'data/mnist/images/'
train_path = data_path + 'train'

image_size = (28, 28)
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
batch_size = 1000
epochs_count = 10

train_data_generator = ImageDataGenerator(validation_split=0.1)

train_directory_iterator = train_data_generator.flow_from_directory(
    train_path,
    target_size=image_size,
    classes=classes,
    batch_size=batch_size,
    color_mode='grayscale',
    subset='training'
)
validation_directory_iterator = train_data_generator.flow_from_directory(
    train_path,
    target_size=image_size,
    classes=classes,
    batch_size=batch_size,
    color_mode='grayscale',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Flatten(),
    Dense(10, activation='softmax'),
])

model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_directory_iterator,
    validation_data=validation_directory_iterator,
    steps_per_epoch=train_directory_iterator.samples / batch_size,
    validation_steps=validation_directory_iterator.samples / batch_size,
    epochs=epochs_count,
    verbose=2
)

if not os.path.exists(model_path):
    os.makedirs(model_path)

model.save(model_path + 'model.h5')
