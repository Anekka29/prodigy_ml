import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
num_samples = 1000
IMAGE_SIZE = 64

X = np.random.rand(num_samples, IMAGE_SIZE * IMAGE_SIZE)  # fake image data
y = np.random.randint(0, 5, num_samples)  # 5 gesture classes

y = to_categorical(y, num_classes=5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = Sequential([
    Dense(128, activation='relu', input_shape=(IMAGE_SIZE * IMAGE_SIZE,)),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("\nEnter random gesture values (simulate):")

sample = np.random.rand(1, IMAGE_SIZE * IMAGE_SIZE)

prediction = model.predict(sample)

print("Predicted Gesture Class:", np.argmax(prediction))

