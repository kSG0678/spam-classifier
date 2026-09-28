import csv
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
messages = []
labels = []

with open("data/SMSSpamCollection", "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter="\t")

    for row in reader:
        if len(row) == 2:
            label, message = row

            messages.append(message)

            if label == "spam":
                labels.append(1)
            else:
                labels.append(0)


print("Dataset loaded successfully!")
print("Total messages:", len(messages))


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# Create model
model = LogisticRegression()


# Train model
model.fit(X_train_vectorized, y_train)


# Test model
predictions = model.predict(X_test_vectorized)


# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)


# Detailed evaluation
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Not Spam", "Spam"]
    )
)


# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully!")
print("Created: model.pkl")
print("Created: vectorizer.pkl")