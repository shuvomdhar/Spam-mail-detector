# 📧 Email Spam Detector

A machine learning-powered web application that classifies emails and SMS messages as spam or ham (legitimate) using Natural Language Processing techniques.

## 🌟 Features

- **Real-time Classification**: Instantly classify email/SMS text as spam or legitimate
- **Machine Learning Model**: Uses Naive Bayes classifier with TF-IDF vectorization
- **Web Interface**: Clean, responsive Flask web application
- **Pre-trained Model**: Ready-to-use trained model included
- **Easy Setup**: Simple installation and deployment process

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, pandas
- **Model**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Frontend**: HTML, CSS
- **Data Storage**: Pickle for model serialization

## 📁 Project Structure

```
spam-detector/
├── .venv/                    # Virtual environment
├── data/
│   └── SMSSpamCollection     # Training dataset
├── models/
│   ├── spam_classifier.pkl   # Trained Naive Bayes model
│   └── tfidf_vectorizer.pkl  # TF-IDF vectorizer
├── static/                   # Static files (if any)
├── templates/
│   └── index.html           # Web interface template
├── .gitignore
├── app.py                   # Flask web application
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── train_model.py         # Model training script
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shuvomdhar/Spam-mail-detector.git
   cd spam-detector
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   Open your browser and go to `http://127.0.0.1:5000`

## 📊 Model Training

If you want to retrain the model with your own data:

1. **Prepare your dataset**
   - Place your dataset in the `data/` folder
   - Ensure it's in the same format as `SMSSpamCollection` (tab-separated with 'ham'/'spam' labels)

2. **Run the training script**
   ```bash
   python train_model.py
   ```

3. **Model performance**
   The script will output accuracy metrics and save the trained model to the `models/` folder.

## 🎯 How It Works

1. **Text Preprocessing**: Input text is cleaned and preprocessed
2. **Feature Extraction**: TF-IDF vectorizer converts text into numerical features
3. **Classification**: Naive Bayes model predicts spam/ham probability
4. **Result Display**: Web interface shows prediction with emoji indicators

### Model Details

- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF with English stop words removal
- **Features**: Maximum 3000 most important words
- **Training Split**: 80% training, 20% testing

## 📝 Usage Examples

### Web Interface
1. Navigate to the web application
2. Paste your email/SMS text in the textarea
3. Click "Check" to get instant classification
4. Results show as:
   - ✅ Ham (legitimate message)
   - 🚨 Spam (spam message)

### Programmatic Usage
```python
import pickle

# Load models
with open("models/spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Classify text
text = "Your message here"
features = vectorizer.transform([text])
prediction = model.predict(features)[0]
result = "Spam" if prediction == 1 else "Ham"
```

## 📋 Requirements

```
blinker==1.9.0
click==8.2.1
colorama==0.4.6
Flask==3.1.2
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.5.2
MarkupSafe==3.0.2
numpy==2.3.2
pandas==2.3.2
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==1.7.1
scipy==1.16.1
six==1.17.0
threadpoolctl==3.6.0
tzdata==2025.2
Werkzeug==3.1.3
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📈 Future Enhancements

- [ ] Support for multiple languages
- [ ] Advanced preprocessing techniques
- [ ] Model comparison dashboard
- [ ] API endpoints for integration
- [ ] Batch processing capabilities
- [ ] Real-time model retraining
- [ ] Performance monitoring dashboard

## 🔍 Dataset Information

The project uses the SMS Spam Collection Dataset, which contains:
- **Total Messages**: ~5,574 SMS messages
- **Ham Messages**: ~4,827 legitimate messages
- **Spam Messages**: ~747 spam messages
- **Format**: Tab-separated values with labels and message text

## 🛡️ Model Performance

Typical performance metrics:
- **Accuracy**: >98%
- **Precision**: High for spam detection
- **Recall**: Balanced for both classes
- **F1-Score**: Optimized for real-world usage

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check existing documentation
- Review the code comments for implementation details

## 📄 License

This project is open source and made for educational purpose.

---

**Made with ❤️ for spam detection and email security**