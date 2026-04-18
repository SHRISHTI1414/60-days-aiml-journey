import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    
    return " ".join(words)

def spam_pipeline(texts):
    cleaned = [preprocess(t) for t in texts]
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(cleaned)
    
    return vectors, vectorizer

# Example
messages = [
    "Congratulations! You have won a free iPhone. Click now!",
    "Hey, are we still meeting tomorrow?"
]

vectors, vectorizer = spam_pipeline(messages)

print(vectors.toarray())
print(vectorizer.get_feature_names_out())