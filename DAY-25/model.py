from sklearn.model_selection import train_test_split, GridSearchCV
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
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])
 
param_grid = {
    "tfidf__max_df": [0.8, 1.0],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__C": [0.1, 1, 10]
}
 
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1")
grid.fit(X_train, y_train)
 
print("Best Parameters:", grid.best_params_)
 
y_pred = grid.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))