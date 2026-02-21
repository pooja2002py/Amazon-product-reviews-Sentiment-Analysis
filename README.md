 Amazon Product Review Sentiment Analysis using NLP
 📌 Project Overview
This project performs Sentiment Analysis on the Amazon Product Reviews dataset using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to classify customer reviews into Positive and Negative sentiment categories.
Sentiment analysis helps businesses extract actionable insights from customer feedback to improve product offerings, marketing strategies, and customer satisfaction.

📂 Dataset
📌 Dataset Source:
🔗 https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews
📌 Description:
A dataset containing Amazon product reviews, including ratings, review text, summaries, and other relevant fields that help determine sentiment.

🛠️ Project Workflow
1️⃣ Data Processing
✔ Handling missing values
✔ Text normalization (lowercasing)
✔ Stopword removal
✔ Tokenization
✔ Lemmatization
✔ Text Vectorization
TF-IDF
Word2Vec
Transformer embeddings (optional)

2️⃣ Handling Imbalanced Classes
Real-world review data often has more positive reviews than negative. Techniques used:
✔ Under-sampling
✔ Over-sampling
✔ SMOTE (Synthetic Minority Over-sampling Technique)

3️⃣ Exploratory Data Analysis (EDA)
✔ Review count per rating
✔ Sentiment distribution visualizations
✔ Word clouds for positive & negative reviews
✔ Top frequent tokens per sentiment class

4️⃣ Model Training
Traditional Machine Learning Models:
✔ Logistic Regression
✔ Naive Bayes
✔ Support Vector Machines
Advanced Models (Optional)
✔ LSTM / BiLSTM
✔ BERT / Transformer-based models

📈 Results & Insights
✔ Sentiment distribution charts
✔ Top positive and negative review words
✔ Performance comparison of models
✔ Impact of sampling techniques

🧾 Sample Visualizations Included
Sentiment Histogram
Word Clouds
Confusion Matrices
ROC Curves
Precision/Recall Bar Charts

💡 Business Impact
Sentiment analysis of Amazon reviews provides:
✔ Data-driven product improvements
✔ Insight into customer response patterns
✔ Better customer experience strategies
✔ Identification of product issues in real-time

🖥️ Technologies Used
Python
Pandas & NumPy
NLTK & SpaCy
Scikit-Learn
TensorFlow / Keras
Matplotlib & Seaborn
WordCloud
imbalanced-learn (SMOTE)
Streamlit (Deployment – Optional)
