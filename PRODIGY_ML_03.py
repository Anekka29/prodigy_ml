
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.DataFrame({
    "fur": [1,1,1,0,0,1,0,1,0,1],
    "tail": [1,1,1,1,0,1,0,1,0,1],
    "legs": [4,4,4,4,2,4,2,4,2,4],
    "bark": [0,0,0,1,1,0,1,0,1,0],
    "type": [0,0,0,1,1,0,1,0,1,0]
})

X = data.drop("type", axis=1)
y = data["type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

print("\nEnter animal features (0 or 1, except legs):")

sample = []
for col in X.columns:
    val = int(input(f"{col}: "))
    sample.append(val)
prediction = model.predict([sample])

if prediction[0] == 0:
    print("Prediction: Cat 🐱")
else:
    print("Prediction: Dog 🐶")


