from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
texts = [
    "I love this product",
    "This is amazing",
    "Worst experience ever",
    "I hate this",
    "It's okay, not bad",
    "Could be better",
    "Absolutely fantastic",
    "Very disappointing"
]

 
labels = [2, 2, 0, 0, 1, 1, 2, 0]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, multi_class="auto"))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))

# Test new inputs
samples = [
    "I really love this",
    "Not good at all",
    "It's fine"
]

preds = pipeline.predict(samples)

mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
for text, pred in zip(samples, preds):
    print(f"{text} → {mapping[pred]}")