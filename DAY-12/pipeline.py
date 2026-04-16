import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

def nlp_pipeline(texts):
    # Step 1: preprocess
    cleaned_texts = [preprocess_text(text) for text in texts]
    
    # Step 2: vectorize
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(cleaned_texts)
    
    return vectors, vectorizer

# Example
docs = [
    "AI is transforming the world",
    "Machine learning is powerful"
]

vectors, vectorizer = nlp_pipeline(docs)

print(vectors.toarray())
print(vectorizer.get_feature_names_out())