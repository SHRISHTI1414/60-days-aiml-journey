from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
 
texts = [
    "I love this movie",
    "This product is amazing",
    "Worst experience ever",
    "I hate this service",
    "Very useful and great",
    "Not worth the money"
]

labels = [1, 1, 0, 0, 1, 0]
 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
 
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)
 
model = LogisticRegression()
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)

 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))