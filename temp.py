from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
    decoders,
)

# 1. Create an empty BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# 2. Convert input into byte-level pieces
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=True
)

# Needed to turn byte-level tokens back into normal text
tokenizer.decoder = decoders.ByteLevel()

# 3. Small training corpus
corpus = [
    "Hello world!",
    "Hello tokenizer!",
    "Hello world!",
    "Hello world!",
]

# 4. Configure and train BPE
trainer = trainers.BpeTrainer(
    vocab_size=300,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

tokenizer.train_from_iterator(corpus, trainer=trainer)

# 5. Encode some text
text = "Hello world!"
encoding = tokenizer.encode(text)

print("Pre-tokenized:")
print(tokenizer.pre_tokenizer.pre_tokenize_str(text))

print("BPE tokens:")
print(encoding.tokens)

print("Token IDs:")
print(encoding.ids)

print("Decoded:")
print(tokenizer.decode(encoding.ids))