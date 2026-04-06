import nltk
import string
import re
from nltk.corpus import stopwords
 
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # lowercase
    text = text.lower()

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    #   (basic split)
    words = text.split()

    # remove stopwords
    clean_words = [word for word in words if word not in stop_words]

    return clean_words


 
text = "AI is transforming industries!!! It is  very fast, powerful, and evolving."

output = clean_text(text)
print("Cleaned Text:", output)