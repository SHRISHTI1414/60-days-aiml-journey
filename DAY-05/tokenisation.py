import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
     
    text = text.lower()

     
    text = re.sub(r'\d+', '', text)

     
    text = text.translate(str.maketrans('', '', string.punctuation))
 # step 4: tokenize
    tokens = word_tokenize(text)

     
    tokens = [word for word in tokens if word not in stop_words]

    return tokens


 
news = "Breaking: AI is transforming global markets in 2025, experts say innovation is accelerating rapidly."

processed = preprocess(news)

print("Processed:", processed)