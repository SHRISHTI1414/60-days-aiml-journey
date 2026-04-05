import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# downloads (run once)
nltk.download('punkt')
nltk.download('wordnet')

# sample product reviews
reviews = [
    "The product was running smoothly and performing well.",
    "I liked the design but the battery was draining quickly.",
    "The features are amazing and the product works perfectly."
]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

for review in reviews:
    print("\nOriginal:", review)

    words = word_tokenize(review.lower())

    stemmed = [stemmer.stem(word) for word in words]
    lemmatized = [lemmatizer.lemmatize(word) for word in words]

    print("Stemmed:", stemmed)
    print("Lemmatized:", lemmatized)