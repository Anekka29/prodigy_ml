import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
food_classes = ["Pizza", "Burger", "Salad", "Pasta", "Rice"]
calories = {
    "Pizza": 285,
    "Burger": 295,
    "Salad": 150,
    "Pasta": 220,
    "Rice": 200
}
num_samples = 1000
IMAGE_SIZE = 64
X = np.random.rand(num_samples, IMAGE_SIZE * IMAGE_SIZE)
y = np.random.randint(0, len(food_classes), num_samples)
y = to_categorical(y, num_classes=len(food_classes))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = Sequential([
    Dense(128, activation='relu', input_shape=(IMAGE_SIZE * IMAGE_SIZE,)),
    Dense(64, activation='relu'),
    Dense(len(food_classes), activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nPredicting food item...")

sample = np.random.rand(1, IMAGE_SIZE * IMAGE_SIZE)
prediction = model.predict(sample)

predicted_class = np.argmax(prediction)
food = food_classes[predicted_class]

print("Predicted Food:", food)
print("Estimated Calories:", calories[food], "kcal")

