from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
 
texts = [
    "I love this product",
    "This is amazing",
    "Worst experience ever",
    "I hate this",
    "Very good quality",
    "Not worth the money"
]

labels = [1, 1, 0, 0, 1, 0]   
 
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])
 
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)
 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
 
sample = ["This product is really good"]
print("Prediction:", model.predict(sample)[0])