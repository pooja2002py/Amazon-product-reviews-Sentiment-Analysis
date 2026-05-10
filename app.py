import streamlit as st 
import pickle 
import re 
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfTransformer

# -----------------------------------
# Download necessary NLTK resources
# -----------------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# -----------------------------------
# Load the saved model and vectorizer
# -----------------------------------

# ❌ REMOVE local system path
# ✅ Use relative path for deployment

with open("naive_bayes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------------
# Stopwords
# -----------------------------------
stop_words = set(stopwords.words("english"))

# -----------------------------------
# Function to preprocess user input
# -----------------------------------
def preprocess_text(text):
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = word_tokenize(text)

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    # Return cleaned text only if it contains at least one valid word
    return " ".join(filtered_words) if filtered_words else None

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.title("🚀 Welcome to Pooja Yadav Sentiment Analysis App! 🎉")

st.subheader("🔍 Amazon Product Review Sentiment Analysis")

st.write(
    "📝 Enter a review, and our model will determine whether it is **Positive** 😊 or **Negative** 😠."
)

user_input = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):

    cleaned_input = preprocess_text(user_input)

    if cleaned_input:

        transformed_input = vectorizer.transform([cleaned_input])

        prediction = model.predict(transformed_input)[0]

        sentiment = (
            "😊 Positive"
            if prediction == "Positive"
            else "😠 Negative"
        )

        st.subheader("Prediction Result:")

        st.write(f"### {sentiment}")

    else:
        st.warning(
            "Please enter a valid review with meaningful words before analyzing."
        )
