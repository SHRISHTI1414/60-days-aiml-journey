from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

 
corpus = [
    "AI is transforming the world",
    "Machine learning is powerful",
    "Tokenization helps models understand text",
    "AI models work on patterns"
]

 
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

 
trainer = WordPieceTrainer(
    vocab_size=200,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

 
tokenizer.train_from_iterator(corpus, trainer)

 
output = tokenizer.encode("AI is powerful")

print("Tokens:", output.tokens)
print("Vocabulary size:", tokenizer.get_vocab_size())