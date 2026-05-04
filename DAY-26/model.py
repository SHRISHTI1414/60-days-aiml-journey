from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
texts = [
    "I love this product",
    "Amazing quality and performance",
    "Worst experience ever",
    "Not worth the price",
    "Really good and useful",
    "I hate this item"
]

labels = [1, 1, 0, 0, 1, 0]
 
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
 
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])
 
pipeline.fit(X_train, y_train)
 
y_pred = pipeline.predict(X_test)
 
print(classification_report(y_test, y_pred))
 
sample = ["This product is very useful"]
print("Prediction:", pipeline.predict(sample)[0])