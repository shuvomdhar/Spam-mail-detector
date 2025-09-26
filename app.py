import os
from flask import Flask, render_template, request
import pickle

# Load trained model + vectorizer
with open("models/spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        email_text = request.form["email_text"]
        features = vectorizer.transform([email_text])
        pred = model.predict(features)[0]
        prediction = "🚨 Spam" if pred == 1 else "✅ Ham"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
