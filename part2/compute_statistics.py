from transformers import T5Tokenizer
import os
import numpy as np

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

def read_lines(path):
    with open(path, "r") as f:
        return [x.strip() for x in f.readlines()]

def tokenize_stats(lines):
    lengths = []
    vocab = set()
    for line in lines:
        ids = tokenizer.encode(line)
        lengths.append(len(ids))
        vocab.update(tokenizer.convert_ids_to_tokens(ids))
    return np.mean(lengths), vocab

def compute(split):
    nl = read_lines(f"data/{split}.nl")
    sql = read_lines(f"data/{split}.sql")

    # Before preprocessing
    nl_len, nl_vocab = tokenize_stats(nl)
    sql_len, sql_vocab = tokenize_stats(sql)

    # After preprocessing (prefix)
    prefixed = [f"translate English to SQL: {x}" for x in nl]
    pre_nl_len, pre_nl_vocab = tokenize_stats(prefixed)
    pre_sql_len, pre_sql_vocab = tokenize_stats(sql)

    print(f"\n=== {split.upper()} ===")
    print("Number of examples:", len(nl))
    print("NL mean length:", nl_len)
    print("SQL mean length:", sql_len)
    print("NL vocabulary size:", len(nl_vocab))
    print("SQL vocabulary size:", len(sql_vocab))

    print("After preprocessing:")
    print("NL mean length (prefixed):", pre_nl_len)
    print("SQL mean length (tokenized):", pre_sql_len)
    print("NL vocab size (prefixed):", len(pre_nl_vocab))
    print("SQL vocab size (prefixed):", len(pre_sql_vocab))


compute("train")
compute("dev")


