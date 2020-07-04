# digit-reader

Approach 1 - `0.9519` accuracy
```
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Flatten(),
    Dense(10, activation='softmax'),
])
```