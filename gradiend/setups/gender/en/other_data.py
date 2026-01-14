import unicodedata
import re
import pandas as pd
import random
from datasets import load_dataset
from collections import defaultdict

# -------------------------------
# 1. Define tokens and groups
# -------------------------------
TOKEN_GROUPS = {
    # pronouns
    "he": "subject_pronoun",
    "she": "subject_pronoun",
    #"him": "object_pronoun",
    #"her": "object_pronoun",
    #"his": "possessive_pronoun",
    #"hers": "possessive_pronoun",

    # person gender singular/plural
    "man": "pearson_gender",
    "woman": "pearson_gender",
    "boy": "person_gender_child",
    "girl": "person_gender_child",

    # family
    "father": "family_parent",
    "mother": "family_parent",
    #"brother": "family_sibling",
    #"sister": "family_sibling",
    "son": "family_child",
    "daughter": "family_child",

    # professions
    #'actor': 'acting',
    #'actress': 'acting',
}

LABEL_MAP = {
    # person
    "man": -1,
    "boy": -1,
    "woman": +1,
    "girl": +1,

    # family
    "father": -1,
    "brother": -1,
    "son": -1,
    "mother": +1,
    "sister": +1,
    "daughter": +1,


    # pronouns
    "he": -1,
    "him": -1,
    "his": -1,
    "she": +1,
    "her": +1,
    "hers": +1,

    # professions
    'actor': -1,
    'actress': +1,
}


TOKENS = list(TOKEN_GROUPS.keys())
token_regex = re.compile(r"\b(" + "|".join(re.escape(t) for t in TOKENS) + r")\b", re.IGNORECASE)

# -------------------------------
# 2. Helper functions
# -------------------------------
def keep_one_token(example):
    return len(find_tokens(example["text"])) == 1



def normalize_text(s):
    # Convert long-s to normal s
    s = s.replace("ſ", "s")

    # NFC normalization (recommended for Wikipedia)
    s = unicodedata.normalize("NFC", s)
    return s

def find_tokens(text):
    text = normalize_text(text)
    return [m.group(0).lower() for m in token_regex.finditer(text)]

def add_metadata(example):
    text = normalize_text(example["text"])
    token = find_tokens(text)[0].lower()
    example["token"] = token
    example["group"] = TOKEN_GROUPS[token]
    example["label"] = LABEL_MAP[token]
    example["masked_text"] = token_regex.sub("[MASK]", text, count=1)
    return example


def balance_dataset(df, max_per_token=1000):
    # remove examples where [MASK] is in the first 10 words
    df = df[df["masked_text"].str.split().map(
        lambda words: "[MASK]" not in words[:10]
    )]

    grouped = df.groupby("token")
    balanced_rows = []

    for token, group in grouped:
        size = len(group)
        if size == 0:
            continue  # skip empty groups

        limit = min(max_per_token, size)
        # sample exactly <= size
        balanced_rows.append(group.sample(n=limit, random_state=42))

    return pd.concat(balanced_rows, ignore_index=True)


# -------------------------------
# 3. Load, filter, add metadata, balance, save
# -------------------------------
def prepare_and_save_balanced_dataset(csv_path="data/other_gender_data_balanced.csv"):
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    filtered = ds.filter(keep_one_token)
    processed = filtered.map(add_metadata)
    df = pd.DataFrame(processed)
    balanced_df = balance_dataset(df)
    balanced_df.to_csv(csv_path, index=False)
    print(len(balanced_df), len(ds))
    return balanced_df

if __name__ == '__main__':
    prepare_and_save_balanced_dataset()