import joblib
import re

# Load saved model and vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    return text


def predict_spam(message):

    cleaned_message = preprocess_text(message)

    vector = vectorizer.transform([cleaned_message])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    if prediction == 1:
        return f"SPAM ({probability[1]:.2%})"
    else:
        return f"HAM ({probability[0]:.2%})"


if __name__ == "__main__":

    message = input("Enter SMS message: ")

    result = predict_spam(message)

    print("\nPrediction:", result)
