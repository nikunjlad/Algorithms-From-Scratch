# Transformers

## Common terms

1. Vocabular size
Refers to the number of tokens we can have our model store are part of its vocabulary. For instance, for a word - "unbelivable", can be broken down into tokens - "un", "beliv" and "able". Like this each word in our vocabulary can be broke into tokens. The vocabulary size says enforces a limit on our model by telling it the budget it needs to have without exceeding it for processing data. Vocabulary can also include special tokens like - <SOS> which is start of sentence or <EOS> which is end of sentence

How to choose vocabulary size?
Choose based on dataset and domain size
- small and special datasets (like medical, legal code): 10_000 - 16_000 - if our text is highly repetitive and does not need massive vocabulary
- standard single language dataset (english, fiction books): choose 32_000 - 50_000. This is the sweet spot for BERT type models and helps account for common words across a language word vocabulary
- large multi-lingual datasets (LLAMA, etc): choose 100_000 to 256_000+ - If model needs to understand multiple codes at once like English, Spanish, etc

Monitor Fertility Rate - Helps check if vocabulary size is good

Fertility Rate = (Number of Tokens) / (Number of words)
- if ratio > 1.5 - vocabulary is too small. tokenizer aggresively chopping words
- if ratio ~= 1.0 - vocabulary too large. Almost every single word kept as whole and not tokenized.
- 1.1 - 1.3 - sweet spot for vocabulary size.

Lastly, choose GPU hardware multipler optimization trick
GPU work well on operations for matrices of size 8, 64, 128.
So, 32_000 is bad choice (single only divisble by 8), but 32_096 and 32_256 (divisible by 64 or 128)

To summarize
1. 8_000 to 16_000 - saves GPU, trains faster - for hobby
2. 32_000 - industry baseline for clean token balance
3. 100_000+ - code-heavy model like Claude - needs room for multiple alphabets/syntax
