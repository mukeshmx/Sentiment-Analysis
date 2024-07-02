import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources if not already downloaded
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load movie reviews dataset
reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

# Shuffle the reviews to mix positive and negative samples
import random
random.shuffle(reviews)

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]
    return ' '.join(cleaned_tokens)

# Preprocess and split the data into features and labels
X = [preprocess_text(' '.join(words)) for words, label in reviews]
y = [label for words, label in reviews]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Function to predict sentiment of a new review
def predict_sentiment(review):
  # Preprocess the user input
  processed_review = preprocess_text(review)
  # Convert the review to a TF-IDF vector
  review_vector = vectorizer.transform([processed_review])
  # Predict sentiment using the trained model
  prediction = clf.predict(review_vector)[0]
  return prediction

# Get user input for the review
user_review = input("Enter your movie review: ")

# Predict sentiment and print the result
sentiment = predict_sentiment(user_review)
if sentiment == 'pos':
  print("Sentiment: Positive Review")
else:
  print("Sentiment: Negative Review")

# Evaluation (optional)
print("\nModel Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
print("\nClassification Report:")
print(classification_report(y_test, clf.predict(X_test)))
