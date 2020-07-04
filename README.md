# digit-reader

Approach 1 - `0.9519` accuracy
```
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Flatten(),
    Dense(10, activation='softmax'),
])
```

Approach 2 - `0.9751` accuracy
```
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Flatten(),
    Dense(24, activation='relu'),
    Dense(10, activation='softmax'),
])
```

Approach 3 - `0.9758` accuracy
```
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Flatten(),
    Dense(24, activation='relu'),
    Dense(10, activation='softmax'),
])
```

Approach 4 - `0.9773` accuracy
```
model = Sequential([
    Conv2D(48, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Flatten(),
    Dense(24, activation='relu'),
    Dense(10, activation='softmax'),
])
```

Approach 5 - `0.9821` accuracy
```
model = Sequential([
    Conv2D(32, (7, 7), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Flatten(),
    Dense(24, activation='relu'),
    Dense(10, activation='softmax'),
])
```

Approach 6 - `0.9819` accuracy
```
model = Sequential([
    Conv2D(48, (7, 7), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Flatten(),
    Dense(24, activation='relu'),
    Dense(10, activation='softmax'),
])
```

Approach 7 - `0.9806` accuracy
```
model = Sequential([
    Conv2D(48, (7, 7), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Flatten(),
    Dense(24, activation='relu'),
    Dense(24, activation='relu'),
    Dense(10, activation='softmax'),
])
```

Approach 8 - `0.9868` accuracy
```
model = Sequential([
    Conv2D(48, (7, 7), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Flatten(),
    Dense(48, activation='relu'),
    Dense(10, activation='softmax'),
])
```

Approach 8 - `0.9847` accuracy
```
train_data_generator = ImageDataGenerator(
    validation_split=0.1,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15
)

model = Sequential([
    Conv2D(48, (7, 7), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Flatten(),
    Dense(48, activation='relu'),
    Dense(10, activation='softmax'),
])
```