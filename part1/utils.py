import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation


    text = example["text"]

    # Tokenize the text
    words = word_tokenize(text)
    new_words = []

    # Probability of replacing a word
    replace_prob = 0.15  # 15%

    for w in words:
        # Only replace alphabetic words
        if w.isalpha() and random.random() < replace_prob:
            synsets = wordnet.synsets(w)

            # Get lemmas (synonyms)
            if synsets:
                lemmas = synsets[0].lemmas()
                if lemmas:
                    synonym = lemmas[0].name().replace("_", " ")

                    # Avoid replacing by itself
                    if synonym.lower() != w.lower():
                        new_words.append(synonym)
                        continue

        # Otherwise keep original
        new_words.append(w)

    # Detokenize
    new_text = TreebankWordDetokenizer().detokenize(new_words)

    example["text"] = new_text
    return example
    ##### YOUR CODE ENDS HERE ######


