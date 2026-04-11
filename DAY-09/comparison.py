from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

text = "AI is transforming the world"

# 1. Whitespace Tokenization
whitespace_tokens = text.split()

# 2. BPE Tokenizer
bpe_tokenizer = Tokenizer(BPE())
bpe_tokenizer.pre_tokenizer = Whitespace()
bpe_trainer = BpeTrainer(vocab_size=50)
bpe_tokenizer.train_from_iterator([text], bpe_trainer)
bpe_output = bpe_tokenizer.encode(text)

# 3. WordPiece Tokenizer
wp_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
wp_tokenizer.pre_tokenizer = Whitespace()
wp_trainer = WordPieceTrainer(vocab_size=50)
wp_tokenizer.train_from_iterator([text], wp_trainer)
wp_output = wp_tokenizer.encode(text)

print("Whitespace:", whitespace_tokens)
print("BPE:", bpe_output.tokens)
print("WordPiece:", wp_output.tokens)