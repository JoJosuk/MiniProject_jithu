import nltk
import string
import pandas
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter

# Data Preprocessing
def preprocess_text(text):

    text = text.lower()

    text = ''.join([c for c in text if c not in string.punctuation])
    
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])

    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])
    
    return text

# Sentiment Analysis
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Topic Modeling
def perform_topic_modeling(reviews):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(reviews)
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf_matrix)
    
    return lda, vectorizer

# Extract Key Phrases
def extract_key_phrases(lda_model, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    key_phrases = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        key_phrases.append([feature_names[i] for i in topic.argsort()[:-6:-1]])
    
    return key_phrases

# Generate Recommendations
def generate_recommendations(key_phrases):
    all_phrases = [phrase for phrases in key_phrases for phrase in phrases]
    phrase_counts = Counter(all_phrases)
    top_phrases = phrase_counts.most_common(5) 
    
    recommendations = []
    
    for phrase, count in top_phrases:
        recommendations.append(f"Recommendation: Improve '{phrase}' (mentioned in {count} reviews)")
    
    return recommendations

# Sample data
'''reviews = [
    "The battery life is too short.",
    "The user interface is confusing.",
    "The product is excellent and highly recommended.",
    "The customer support was unhelpful.",
    "The price is too high for the features offered.",
    "The battery drains quickly.",
    "The software crashes frequently."
]'''

reviews=pandas.read_csv("az-reviews.csv")
print(reviews)

# Preprocess 
preprocessed_reviews = [preprocess_text(review) for review in reviews]

# sentiment analysis
sentiments = [analyze_sentiment(review) for review in preprocessed_reviews]

# topic modeling
lda_model, vectorizer = perform_topic_modeling(preprocessed_reviews)

# Extract key phrases
key_phrases = extract_key_phrases(lda_model, vectorizer)


# recommendations
recommendations = generate_recommendations(key_phrases)

# output
for i, review in enumerate(reviews):
    print(f"Review {i+1}:")
    print("Sentiment:", sentiments[i])
    print("Key Phrases:", key_phrases[i])
    print()

print("Recommendations:")
for recommendation in recommendations:
    print(recommendation)
