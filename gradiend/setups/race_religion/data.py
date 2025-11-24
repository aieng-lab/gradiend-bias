import os
import argparse
import json
from itertools import permutations

import tqdm
from datasets import load_dataset, Dataset

def _create_bias_attribute_words(path, bias_type):
    """Load attribute words from JSON, select subset for augmentation type."""
    with open(path, "r", encoding="utf-8") as f:
        bias_words = json.load(f)
    return bias_words[bias_type]

def match_casing(source: str, target: str) -> str:
    """Adjust casing of target word to match source word."""
    if source.isupper():
        return target.upper()
    elif source.islower():
        return target.lower()
    elif source[0].isupper() and source[1:].islower():
        return target.capitalize()
    else:
        # mixed/other casing: just return target as-is
        return target


def get_base_terms(bias_attribute_words):
    base_terms = bias_attribute_words[0]
    term_map = {'caucasian': 'white'}
    base_terms = [term_map.get(t, t) for t in base_terms]
    return base_terms

def pairwise_counterfactual_datasets(dataset, bias_attribute_words):
    replacement_pairs = []
    for group in bias_attribute_words:
        for orig, repl in permutations(group, 2):
            replacement_pairs.append((orig, repl))

    base_terms = get_base_terms(bias_attribute_words)
    base_term_mapping = {t[i]: base_terms[i] for t in bias_attribute_words for i in range(len(t))}
    outputs = {pair: [] for pair in replacement_pairs}

    for text in tqdm.tqdm(dataset["text"], desc="Processing dataset for counterfactual pairs"):
        words = text.split()  # preserve original casing
        lower_words = [w.lower() for w in words]

        for (orig, repl) in replacement_pairs:
            orig_lower, repl_lower = orig.lower(), repl.lower()

            if orig_lower in lower_words and lower_words.count(orig_lower) == 1:
                swapped, masked, sources, targets = [], [], [], []
                for w in words:
                    if w.lower() == orig_lower:
                        swapped_word = match_casing(w, repl)
                        swapped.append(swapped_word)
                        masked.append("[MASK]")
                        source = w  # original word with its casing
                        target = swapped_word  # replacement with matched casing
                    else:
                        swapped.append(w)
                        masked.append(w)

                base_orig = base_term_mapping[orig_lower]
                base_repl = base_term_mapping[repl_lower]

                outputs[(base_orig, base_repl)].append(
                    (text, " ".join(swapped), " ".join(masked), source, target)
                )

    datasets_per_pair = {}
    for (orig, repl), results in outputs.items():
        if results:
            texts = [t[0] for t in results]
            swapped = [t[1] for t in results]
            masked = [t[2] for t in results]
            sources = [t[3] for t in results]
            targets = [t[4] for t in results]
            n = len(texts)
            print(f"Creating dataset for {orig} → {repl} with {n} examples")

            datasets_per_pair[(orig, repl)] = Dataset.from_dict({
                "text": texts,
                "swapped": swapped,
                "masked": masked,
                "source": sources,
                "target": targets,
                "source_id": [orig] * n,
                "target_id": [repl] * n,
            })

    return datasets_per_pair

def generate_data(bias_type='race'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default='data/text/wikipedia-10.txt')
    parser.add_argument("--output_dir", type=str, default=f'data/{bias_type}')
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("text", data_files={"train": args.train_file})["train"]

    # Load bias attribute words
    bias_attribute_words = _create_bias_attribute_words(
        f"data/bias_attribute_words.json",
        bias_type=bias_type,
    )

    # Pairwise augmentation
    pairwise_datasets = pairwise_counterfactual_datasets(dataset, bias_attribute_words)

    for (orig, repl), ds in pairwise_datasets.items():
        name = f"{orig.lower()}_to_{repl.lower()}"
        save_path = os.path.join(args.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        ds.save_to_disk(save_path)
        print(f"Saved dataset for {orig} → {repl} at {save_path}")


def generate_race_data():
    generate_data(bias_type='race')

def generate_religion_data():
    generate_data(bias_type='religion')

if __name__ == "__main__":
    generate_race_data()
    generate_religion_data()
