import json
import os
import random
import time
from pprint import pprint

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from torch import softmax
from tqdm import tqdm

from transformers import AutoModelForMaskedLM


from gradiend.model import ModelWithGradiend, AutoModelForLM, AutoTokenizerForLM, InstructTokenizerWrapper
from gradiend.util import hash_model_weights, normalization, default_accuracy_function, init_matplotlib


import torch
import pandas as pd
from collections import defaultdict


import torch
import numpy as np
from collections import defaultdict

import torch
import numpy as np
from collections import defaultdict

def compute_bias_score_v1(model,
                       tokenizer,
                       df,
                       targets,  # dict of key_stratify values -> list of target words
                       key_text='masked',
                       ):
    """
    Compute bias score per-example, then average.
    Each example produces:
        - similarity across groups
        - magnitude of target probability
        - combined score = similarity * magnitude
    Finally, results are averaged across all examples.
    """
    model.eval()
    device = model.device
    is_generative = 'ForMaskedLM' not in model.__class__.__name__
    is_llama = 'llama' in model.name_or_path.lower()

    def create_token_ids(words):
        ids = []
        if is_llama:
            words += [f' {w.strip()}' for w in words]

        for w in set(words):
            tokenized = tokenizer.tokenize(w)
            if len(tokenized) != 1:
                print(f"Warning: word '{w}' tokenized into multiple tokens {tokenized};")
                continue
            token_id = tokenizer.convert_tokens_to_ids(tokenized[0])
            ids.append(token_id)
        return list(set(ids))

    # Convert target words to token ids
    targets_ids = {group: create_token_ids(words) for group, words in targets.items()}

    all_scores, all_sims, all_mags = [], [], []
    group_probs_all = defaultdict(list)

    with torch.no_grad():
        for _, row in df.iterrows():
            if is_generative:
                prefix = row[key_text].split('[MASK]')[0]

                if not prefix.strip():
                    continue

                # Tokenize only prefix, since suffix isn’t used for next-token prediction
                inputs = tokenizer(prefix, return_tensors="pt", max_length=512, truncation=True).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                # Get logits for the next token after the prefix
                logits = outputs.logits[0, -1, :]  # last position
                probs = torch.softmax(logits, dim=-1)
            else:
                text = row[key_text].replace('[MASK]', tokenizer.mask_token)

                inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
                mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                if len(mask_idx) == 0:
                    continue
                mask_idx = mask_idx.item()

                outputs = model(**inputs)
                logits = outputs.logits[0, mask_idx, :]
                probs = torch.softmax(logits, dim=-1)

            # Collect probs for all groups for THIS example
            per_group_probs = {}
            for group, candidate_ids in targets_ids.items():
                if not candidate_ids:
                    continue

                candidate_prob = probs[candidate_ids].sum().cpu().item()
                per_group_probs[group] = candidate_prob
                group_probs_all[group].append(candidate_prob)

            if len(per_group_probs) < 2:
                continue  # need at least 2 groups to compare

            # --- Compute similarity across groups ---
            values = np.array(list(per_group_probs.values()))
            all_pairs = []
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    all_pairs.append((values[i], values[j]))
            l1_diff = np.max([abs(a - b) for a, b in all_pairs])
            similarity = 1.0 - l1_diff

            # --- Magnitude = smallest group prob ---
            magnitude = float(np.min(values))

            # --- Score ---
            score = similarity * magnitude

            all_sims.append(similarity)
            all_mags.append(magnitude)
            all_scores.append(score)

        if not group_probs_all:
            raise ValueError("No valid group probabilities computed; check your data and targets.")
        # Final aggregation
        group_means = {g: float(np.mean(v)) for g, v in group_probs_all.items() if v}
        magnitude = float(np.mean(all_mags)) if all_mags else 0.0
        similarity = float(np.mean(all_sims)) if all_sims else 0.0
        score = float(np.mean(all_scores)) if all_scores else 0.0

        # Print after loop (like original)
        print(group_means, 'mag', magnitude, 'sim', similarity, 'score', score)

        return {
            'score': score,
            'similarity': similarity,
            'magnitude': magnitude,
            'group_probs': group_means,
        }

import torch
import numpy as np
from collections import defaultdict

def compute_bias_score_v2(model,
                       tokenizer,
                       df,
                       targets,  # dict of key_stratify values -> list of target words
                       key_text='masked',
                       batch_size=16,
                       ):
    """
    Compute bias score per-example, then average.
    Each example produces:
        - similarity across groups
        - magnitude of target probability
        - combined score = similarity * magnitude
    Finally, results are averaged across all examples.
    """
    model.eval()
    device = model.device
    is_generative = 'ForMaskedLM' not in model.__class__.__name__
    is_llama = 'llama' in model.name_or_path.lower()

    # --- Precompute target token ids ---
    def create_token_ids(words):
        _tokenizer = tokenizer
        if hasattr(tokenizer, 'tokenizer'):
            _tokenizer = tokenizer.tokenizer
        toks = []
        if is_llama:
            words = words + [f' {w.strip()}' for w in words]
        for w in set(words):
            tokenized = _tokenizer.tokenize(w)
            if len(tokenized) == 1:
                toks.append(_tokenizer.convert_tokens_to_ids(tokenized[0]))
            #elif not is_llama:
            #    continue
            else:
                # check if the first token is at most 50% of the word
                first_token = tokenized[0]
                first_token_str = _tokenizer.convert_tokens_to_string([first_token]).strip()
                if len(first_token_str) / len(w) < 0.5:
                    print(f"Warning: word '{w}' tokenized into multiple tokens {tokenized}; first token '{first_token_str}' is less than 50% of the word.")
                else:
                    print(f"Warning: word '{w}' tokenized into multiple tokens {tokenized}; using first token only.")
                    toks.append(_tokenizer.convert_tokens_to_ids(tokenized[0]))
        return list(set(toks))

    targets_ids = {g: create_token_ids(ws) for g, ws in targets.items()}

    # ensure that each target has at least one token id
    for g, ids in targets_ids.items():
        if not ids:
            raise ValueError(f"Target group '{g}' has no valid token ids; check your target words and tokenizer.")

    all_scores, all_sims, all_mags = [], [], []
    group_probs_all = defaultdict(list)

    rows = df.to_dict("records")

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start+batch_size]

            if is_generative:
                # Prefix only (drop empty prefixes)
                prefixes = [r[key_text].split('[MASK]')[0] for r in batch]
                valid = [(i, p) for i, p in enumerate(prefixes) if p.strip()]
                if not valid:
                    continue

                idxs, prefix_texts = zip(*valid)
                inputs = tokenizer(list(prefix_texts), return_tensors="pt",
                                   padding=True, truncation=True, max_length=512).to(device)
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]  # next-token logits
                probs = torch.softmax(logits, dim=-1)
                example_probs = {i: probs[j] for j, i in enumerate(idxs)}

            else:  # MaskedLM
                texts = [r[key_text].replace('[MASK]', tokenizer.mask_token) for r in batch]
                inputs = tokenizer(list(texts), return_tensors="pt",
                                   padding=True, truncation=True, max_length=512).to(device)

                mask_idxs = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
                if len(mask_idxs) == 0:
                    continue

                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

                # Map each example to its masked position’s probs
                example_probs = {}
                for b_idx, pos in mask_idxs.tolist():
                    example_probs[b_idx] = probs[b_idx, pos, :]

            # --- Process each example ---
            for b_idx, row in enumerate(batch):
                if b_idx not in example_probs:
                    continue

                p = example_probs[b_idx]
                per_group_probs = {}
                for g, ids in targets_ids.items():
                    if ids:
                        per_group_probs[g] = float(p[ids].sum())
                        group_probs_all[g].append(per_group_probs[g])

                if len(per_group_probs) < 2:
                    continue

                values = np.array(list(per_group_probs.values()))
                # max |a-b| = max - min
                l1_diff = values.max() - values.min()
                similarity = 1.0 - l1_diff
                magnitude = float(values.min())
                score = similarity * magnitude

                all_sims.append(similarity)
                all_mags.append(magnitude)
                all_scores.append(score)

    if not group_probs_all:
        raise ValueError("No valid group probabilities computed; check your data and targets.")

    # Final aggregation
    group_means = {g: float(np.mean(v)) for g, v in group_probs_all.items() if v}
    magnitude = float(np.mean(all_mags)) if all_mags else 0.0
    similarity = float(np.mean(all_sims)) if all_sims else 0.0
    score = float(np.mean(all_scores)) if all_scores else 0.0

    print(group_means, 'mag', magnitude, 'sim', similarity, 'score', score)

    return {
        'score': score,
        'similarity': similarity,
        'magnitude': magnitude,
        'group_probs': group_means,
    }

token_cache = {}

def compute_bias_score(model,
                       tokenizer,
                       df,
                       targets,  # dict of key_stratify values -> list of target words/phrases
                       key_text='masked',
                       batch_size=16,
                       ):
    """
    Compute bias score per-example, then average.
    Supports multi-token targets for MaskedLM.
    Each example produces:
        - similarity across groups
        - magnitude of target probability
        - combined score = similarity * magnitude
    Finally, results are averaged across all examples.
    """


    model.eval()
    device = model.device
    is_generative = 'ForMaskedLM' not in model.__class__.__name__
    if is_generative:
        return compute_bias_score_v2(model, tokenizer, df, targets, key_text, batch_size)

    # --- Precompute target token ids (support multi-token) ---
    def create_token_ids(words):

        cache_id = (tokenizer.name_or_path, hash(tuple(sorted(words))))
        if cache_id in token_cache:
            return token_cache[cache_id]

        _tokenizer = tokenizer
        if hasattr(tokenizer, 'tokenizer'):
            _tokenizer = tokenizer.tokenizer
        result = [ _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(w)) for w in set(words) ]
        token_cache[cache_id] = result
        return result


    targets_ids = {g: create_token_ids(ws) for g, ws in targets.items()}

    for g, id_lists in targets_ids.items():
        if not id_lists:
            raise ValueError(f"Target group '{g}' has no valid token ids; check your targets.")

    all_scores, all_sims, all_mags = [], [], []
    group_probs_all = defaultdict(list)

    rows = df.to_dict("records")

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start+batch_size]

            if is_generative:
                # prefix-only mode
                prefixes = [r[key_text].split('[MASK]')[0] for r in batch]
                valid = [(i, p) for i, p in enumerate(prefixes) if p.strip()]
                if not valid:
                    continue

                idxs, prefix_texts = zip(*valid)
                inputs = tokenizer(list(prefix_texts), return_tensors="pt",
                                   padding=True, truncation=True, max_length=512).to(device)
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                example_probs = {i: probs[j] for j, i in enumerate(idxs)}

                # --- Process each example ---
                for b_idx, row in enumerate(batch):
                    if b_idx not in example_probs:
                        continue

                    p = example_probs[b_idx]
                    per_group_probs = {}
                    for g, ids in targets_ids.items():
                        if ids:
                            per_group_probs[g] = float(p)
                            group_probs_all[g].append(per_group_probs[g])

                    if len(per_group_probs) < 2:
                        continue

                    values = np.array(list(per_group_probs.values()))
                    # max |a-b| = max - min
                    l1_diff = values.max() - values.min()
                    similarity = 1.0 - l1_diff
                    magnitude = float(values.min())
                    score = similarity * magnitude

                    all_sims.append(similarity)
                    all_mags.append(magnitude)
                    all_scores.append(score)

            else:  # MaskedLM with multi-token support
                texts = [r[key_text] for r in batch]

                # group all targets by length once
                targets_by_len = defaultdict(list)
                for g, id_lists in targets_ids.items():
                    for ids in id_lists:
                        targets_by_len[len(ids)].append((g, ids))

                # for each length, build expanded texts for *all examples* and run in batch
                example_probs = defaultdict(dict)  # b_idx -> {group: prob}
                for k, g_and_ids in targets_by_len.items():
                    expanded_texts = []
                    example_map = []  # map back expanded idx -> original example
                    for b_idx, text in enumerate(texts):
                        if "[MASK]" not in text:
                            continue
                        masked_text = text.replace("[MASK]", " ".join([tokenizer.mask_token] * k))
                        expanded_texts.append(masked_text)
                        example_map.append(b_idx)

                    if not expanded_texts:
                        continue

                    inputs = tokenizer(expanded_texts, return_tensors="pt",
                                       padding=True, truncation=True, max_length=512).to(device)
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)

                    # for each expanded input, score all targets of this length
                    for local_idx, b_idx in enumerate(example_map):
                        mask_positions = (inputs["input_ids"][local_idx] == tokenizer.mask_token_id).nonzero(
                            as_tuple=False).squeeze(-1).tolist()

                        for g, ids in g_and_ids:
                            if len(ids) > len(mask_positions):
                                continue
                            logp = 0.0
                            for j, tok_id in enumerate(ids):
                                pos = mask_positions[j]
                                logp += torch.log(probs[local_idx, pos, tok_id] + 1e-12)
                            p = torch.exp(logp).item()

                            example_probs[b_idx][g] = example_probs[b_idx].get(g, 0.0) + p
                            group_probs_all[g].append(p)

                # compute per-example scores
                for b_idx, row in enumerate(batch):
                    per_group_probs = example_probs.get(b_idx, {})
                    if len(per_group_probs) < 2:
                        continue

                    values = np.array(list(per_group_probs.values()))
                    l1_diff = values.max() - values.min()
                    similarity = 1.0 - l1_diff
                    magnitude = float(values.min())
                    score = similarity * magnitude

                    all_sims.append(similarity)
                    all_mags.append(magnitude)
                    all_scores.append(score)

    if not group_probs_all:
        raise ValueError("No valid group probabilities computed; check your data and targets.")

    group_means = {g: float(np.mean(v)) for g, v in group_probs_all.items() if v}
    magnitude = float(np.mean(all_mags)) if all_mags else 0.0
    similarity = float(np.mean(all_sims)) if all_sims else 0.0
    score = float(np.mean(all_scores)) if all_scores else 0.0

    print(group_means, 'mag', magnitude, 'sim', similarity, 'score', score)

    return {
        'score': score,
        'similarity': similarity,
        'magnitude': magnitude,
        'group_probs': group_means,
    }




def compute_lms(model, tokenizer, texts, ignore):
    from gradiend.evaluation.mlm import evaluate_mlm, evaluate_clm_perplexity
    is_generative = tokenizer.mask_token is None
    if is_generative:
        is_llama = 'llama' in model.__class__.__name__.lower()
        batch_size = 8 if is_llama else 32
        result = evaluate_clm_perplexity(model, tokenizer, texts[:1000], verbose=False, ignore=ignore, batch_size=batch_size)
    else:
        result, stats = evaluate_mlm(model, tokenizer, texts, verbose=False, ignore=ignore)
    return result



def calculate_average_probability_difference(fairness_dict):
    total_diff = 0.0
    num_texts = len(fairness_dict)

    for text, probs in fairness_dict.items():
        prob_m = probs['M']
        prob_f = probs['F']
        total_diff += abs(prob_m - prob_f)


    total_diff /= num_texts
    return total_diff


def calculate_average_prediction_quality(gender_probabilities):
    num_texts = len(gender_probabilities)

    keys = list(gender_probabilities.values())[0]['M'] # todo deprecate
    if isinstance(keys, dict):
        total_sums = {key: sum([sum([v[key] for v in probs.values() if isinstance(v, dict)]) for probs in gender_probabilities.values()]) for key in keys}
        averages = {key: total_sums[key] / num_texts for key in keys}
    else:
        total_sum = sum([sum(v for v in probs.values() if isinstance(v, float)) for probs in gender_probabilities.values()])
        averages = total_sum / num_texts
    return averages


def calculate_baseline_change(current_model, baseline_model, type='total'):
    gender_preference_accuracy = 0
    overall_accuracy = 0
    absolute_difference_sum = 0

    total_texts = len(current_model)

    for text in current_model:
        current_M = current_model[text]['M'][type]
        current_F = current_model[text]['F'][type]
        baseline_M = baseline_model[text]['M'][type]
        baseline_F = baseline_model[text]['F'][type]

        # Gender Preference Accuracy
        current_prefers_M = current_M > current_F
        baseline_prefers_M = baseline_M > baseline_F
        if current_prefers_M == baseline_prefers_M:
            gender_preference_accuracy += 1

        # Overall Accuracy
        current_max_gender = 'M' if current_M > current_F else 'F'
        baseline_max_gender = 'M' if baseline_M > baseline_F else 'F'
        if current_max_gender == baseline_max_gender:
            overall_accuracy += 1

        # Absolute Difference in Preference
        current_diff = current_M - current_F
        baseline_diff = baseline_M - baseline_F
        absolute_difference_sum += abs(current_diff - baseline_diff)

    metrics = {
        "gender_preference_accuracy": gender_preference_accuracy / total_texts,
        "overall_accuracy": overall_accuracy / total_texts,
        "average_absolute_difference": absolute_difference_sum / total_texts
    }

    return metrics


def compute_gender_preference_accuracy(current_model: dict) -> float:
    """
    Computes the accuracy of how often the current model prefers the male gender over the female gender.
    """
    male_preference_count = 0
    total_texts = len(current_model)

    for text in current_model:
        if isinstance(current_model[text]['M'], dict):
            current_M = current_model[text]['M']['total']
            current_F = current_model[text]['F']['total']
        else:
            current_M = current_model[text]['M']
            current_F = current_model[text]['F']

        if current_M > current_F:
            male_preference_count += 1

    return male_preference_count / total_texts

token_indices_cache = {}
gender_mapping_cache = {}

def evaluate_gender_bias_name_predictions(model, tokenizer, text_prefix=None, batch_size=64, df_name=None, baseline=None, verbose=False):
    from gradiend.setups.gender.en.data import read_geneutral, read_gentypes, read_namextend, read_namexact

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    is_instruction_model = isinstance(tokenizer, InstructTokenizerWrapper)
    is_generative = tokenizer.mask_token is None
    if is_generative:
        tokenizer.pad_token = tokenizer.eos_token

    if is_instruction_model:
        tokenizer.system_prompt = tokenizer.system_prompt_name

    data = read_gentypes()
    names_df = read_namextend()
    def preprocess(text):
        return text.lower().replace('ġ', '').replace('Ġ', '').strip()

    start = time.time()
    tokenizer_names_id = (tokenizer.name_or_path, hash(tuple(names_df['name'].tolist())))
    if tokenizer_names_id in gender_mapping_cache:
        gender_mapping_he, gender_mapping_she = gender_mapping_cache[tokenizer_names_id]
    else:
        names_df['name_lower'] = names_df['name'].str.lower()
        gender_mapping_he = names_df[names_df['gender'] == 'M'].set_index('name_lower')['prob_M'].to_dict()
        gender_mapping_she = names_df[names_df['gender'] == 'F'].set_index('name_lower')['prob_F'].to_dict()

        tokenizer_vocab_lower = [preprocess(k) for k in tokenizer.vocab.keys()]
        gender_mapping_he = {k: v for k, v in gender_mapping_he.items() if k.lower() in tokenizer_vocab_lower}
        gender_mapping_she = {k: v for k, v in gender_mapping_she.items() if k.lower() in tokenizer_vocab_lower}
        print(f"Preprocessing names took {time.time() - start:.2f} seconds")
        gender_mapping_cache[tokenizer_names_id] = (gender_mapping_he, gender_mapping_she)

    if text_prefix is False:
        # the prefix "My Friend, " is removed, i.e., also are suitable gendered predictions
        gender_mapping_he['he'] = 1.0
        gender_mapping_she['she'] = 1.0

        if 'He' in tokenizer.vocab:
            gender_mapping_he['He'] = 1.0
        if 'She' in tokenizer.vocab:
            gender_mapping_she['She'] = 1.0


    if tokenizer.name_or_path in token_indices_cache:
        he_token_indices, she_token_indices, he_token_factors, she_token_factors = token_indices_cache[tokenizer.name_or_path]
    else:
        start = time.time()

        he_tokens = [name for name in tokenizer.vocab if preprocess(name) in gender_mapping_he]
        he_token_factors = np.array([gender_mapping_he[preprocess(name)] for name in he_tokens])

        she_tokens = [name for name in tokenizer.vocab if preprocess(name) in gender_mapping_she]
        she_token_factors = np.array([gender_mapping_she[preprocess(name.lower())] for name in she_tokens])

        he_token_indices = [tokenizer.vocab[name] for name in he_tokens]
        she_token_indices = [tokenizer.vocab[name] for name in she_tokens]
        token_indices_cache[tokenizer.name_or_path] = (he_token_indices, she_token_indices, he_token_factors, she_token_factors)
        end = time.time()
        print(f"Token indices took {end - start:.2f} seconds")

    gender_probabilities = {}
    gender_data = {key: [] for key in
                   ['text', 'gender', 'token', 'probability', 'prob_he', 'prob_she', 'most_likely_token', 'split']}

    all_texts = []
    for _, record in data.iterrows():
        text = record['text']
        if is_generative and not is_instruction_model:
            text = f'The person, who {text.removeprefix("My friend, [NAME],").removesuffix(".")}, has the first name'
        elif text_prefix:
            if text.startswith('My friend, [NAME],'):
                text = f'{text_prefix.strip()}, [NAME], {text.removeprefix("My friend, [NAME],").strip()}'
            elif text.startswith('[NAME]'):
                text = f'{text_prefix.strip()}, [NAME], {text.removeprefix("[NAME]").strip()}'
            else:
                text = f'{text_prefix.strip()} {text}'
        elif text_prefix is False:
            text = text.removeprefix('My friend, [NAME],').strip()
        if is_instruction_model or is_generative:
            masked_text = text
        else:
            masked_text = text.replace("[NAME]", tokenizer.mask_token)
        all_texts.append(masked_text)

    vocab = {v: k for k, v in tokenizer.vocab.items()}

    # token indices for he & she (independent of casing)
    token_idx_he = tokenizer.vocab['he']
    token_idx_she = tokenizer.vocab['she']

    for start_idx in range(0, len(all_texts), batch_size):
        end_idx = min(start_idx + batch_size, len(all_texts))
        batch_texts = all_texts[start_idx:end_idx]

        # Tokenize the batch
        batch_tokenized_text = tokenizer(batch_texts, padding=True, return_tensors="pt", truncation=True)
        input_ids = batch_tokenized_text["input_ids"].to(device)
        attention_mask = batch_tokenized_text["attention_mask"].to(device)

        # Find mask token index in each input
        if is_instruction_model:
            #mask_token_index = (input_ids.squeeze() != tokenizer.pad_token_id).nonzero()[-1].item() - 1

            mask_token_index = (input_ids.squeeze() != tokenizer.pad_token_id).sum(dim=1) + 1

            #pad_id = tokenizer.pad_token_id
            #last_non_pad_indices = (input_ids != pad_id).int().flip(dims=[1]).argmax(dim=1)
            #last_non_pad_indices = input_ids.size(1) - 1 - last_non_pad_indices
            #mask_token_index = last_non_pad_indices

        elif is_generative:
            mask_token_index = (input_ids.squeeze() != tokenizer.pad_token_id).sum(dim=1) - 1
        else:
            mask_token_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

        for i in range(len(batch_texts)):
            text = batch_texts[i]
            text_dict = {key: {} for key in {'M', 'F'}}
            masked_index = mask_token_index[i].item()
            predictions = logits[i, masked_index]

            softmax_probs = softmax(predictions, dim=-1).cpu()

            most_likely_token_id = torch.argmax(softmax_probs).item()
            most_likely_token = tokenizer.decode([most_likely_token_id])
            prob_he = softmax_probs[token_idx_he].item()
            prob_she = softmax_probs[token_idx_she].item()

            if verbose:
                # print the top 100 tokens
                top_k = 100
                top_k_values, top_k_indices = torch.topk(softmax_probs, top_k)
                top_k_tokens = [vocab[int(i)] for i in top_k_indices]
                top_k_probs = top_k_values.tolist()
                print(f"Top {top_k} tokens:")
                for token, prob in zip(top_k_tokens, top_k_probs):
                    print(f"{token}: {prob:.4f}")

            for gender, token_indices, token_factors in [('M', he_token_indices, he_token_factors), ('F', she_token_indices, she_token_factors)]:

                gender_probs = softmax_probs[token_indices] * token_factors

                relevant_token_indices = token_indices # [i for i, prob in zip(token_indices, gender_probs) if prob > 0.0001]
                tokens = [vocab[int(i)] for i in relevant_token_indices]

                n = len(tokens)
                if df_name:
                    gender_data['text'] += n * [text]
                    gender_data['gender'] += n * [gender]
                    gender_data['token'] += tokens
                    gender_data['probability'] += gender_probs.tolist()
                    gender_data['prob_he'] += n * [prob_he]
                    gender_data['prob_she'] += n * [prob_she]
                    gender_data['most_likely_token'] += n * [most_likely_token]

                gender_prob = gender_probs.sum().item()
                text_dict[gender] = gender_prob

            if isinstance(text_dict[gender], dict): # todo deprecate?
                keys = text_dict[gender].keys()
                sums = {key: sum(text_dict[g][key] for g in text_dict) for key in keys}

                factor_M = {key: text_dict['M'][key] / sums[key] if sums[key] > 0 else 0 for key in keys}
                factor_F = {key: text_dict['F'][key] / sums[key] if sums[key] > 0 else 0 for key in keys}
                factor_max = {key: max(factor_M[key], factor_F[key]) for key in keys}
                sum_M = text_dict['M']['total']
                sum_F = text_dict['F']['total']
            else:
                total_sum = sum(text_dict.values())
                factor_M = text_dict['M'] / total_sum if total_sum > 0 else 0
                factor_F = text_dict['F'] / total_sum if total_sum > 0 else 0
                factor_max = max(factor_M, factor_F)
                sum_M = text_dict['M']
                sum_F = text_dict['F']

            text_dict['factor_M'] = factor_M
            text_dict['factor_F'] = factor_F
            text_dict['factor_max'] = factor_max

            text_apd = min(1.0, max(0.0, abs(sum_M - sum_F)))

            if not (0 <= text_apd <= 1):
                raise ValueError(f"Invalid APD: {text_apd}")

            text_dict['text_apd'] = text_apd
            text_dict['text_bpi'] = (1 - text_apd) * (sum_M + sum_F)
            text_dict['text_mpi'] = (1 - sum_F) * sum_M
            text_dict['text_fpi'] = (1 - sum_M) * sum_F

            if not (0 <= text_dict['text_bpi'] <= 1):
                raise ValueError(f"Invalid BPI: {text_dict['text_bpi']}")

            gender_probabilities[text] = text_dict

    apd = calculate_average_probability_difference(gender_probabilities)
    _bpi = np.mean([prob['text_bpi'] for prob in gender_probabilities.values()]).item()
    _mpi = np.mean([prob['text_mpi'] for prob in gender_probabilities.values()]).item()
    _fpi = np.mean([prob['text_fpi'] for prob in gender_probabilities.values()]).item()

    prediction_quality = calculate_average_prediction_quality(gender_probabilities)
    if isinstance(baseline, dict):
        baseline_change = calculate_baseline_change(gender_probabilities, baseline)
    else:
        baseline_change = None

    if df_name:
        df = pd.DataFrame.from_dict(gender_data)
        file_name = f'results/cache/metrics/models/{df_name.removeprefix("results/").removeprefix("models/").removeprefix("cache/").removesuffix(".csv")}.csv'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        df.to_csv(file_name, index=False)

    # Calculate average probabilities for each gender
    keys = list(gender_probabilities.values())[0]['M']
    if isinstance(keys, dict):
        avg_prob_m = {key: sum(probs['M'][key] for probs in gender_probabilities.values()) / len(gender_probabilities) for key in keys}
        avg_prob_f = {key: sum(probs['F'][key] for probs in gender_probabilities.values()) / len(gender_probabilities) for key in keys}
        preference_score = {key: abs(avg_prob_m[key] - avg_prob_f[key]) for key in keys}
    else:
        avg_prob_m = sum(probs['M'] for probs in gender_probabilities.values()) / len(gender_probabilities)
        avg_prob_f = sum(probs['F'] for probs in gender_probabilities.values()) / len(gender_probabilities)
        preference_score = abs(avg_prob_m - avg_prob_f)

    he_prob = compute_gender_preference_accuracy(gender_probabilities)

    print(f'P(M)= {avg_prob_m:.4f}, P(F)={avg_prob_f:.4f}, APD={apd:.4f}, BPI={_bpi:.4f}, MPI={_mpi:.4f}, FPI={_fpi:.4f}')

    result = {
        'apd': apd,
        'pq': prediction_quality,
        '_bpi': _bpi,
        '_mpi': _mpi,
        '_fpi': _fpi,
        'avg_prob_m': avg_prob_m,
        'avg_prob_f': avg_prob_f,
        'preference_score': preference_score,
        'he_prob': he_prob,
    }

    if baseline is True:
        result['baseline_change'] = baseline_change
        return result, gender_probabilities

    return result

# todo currently not used but may be useful in the future
def evaluate_gender_bias_multitoken(
    model,
    tokenizer,
    text_prefix=None,
    baseline=None,
):
    """
    Same interface & output as `evaluate_gender_bias_name_predictions`,
    but supports multi-token names with exact sequence probability.
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    #model.eval()

    is_instruction_model = hasattr(tokenizer, 'system_prompt')
    is_generative = tokenizer.mask_token is None
    if is_generative:
        tokenizer.pad_token = tokenizer.eos_token

    # load data
    data = read_gentypes()
    names_df = read_namextend()

    # build name -> ids map and gender factors
    names_df['ids'] = names_df['name'].apply(
        lambda nm: tokenizer(nm, add_special_tokens=False).input_ids
    )
    # lowercase mapping
    names_df['name_lower'] = names_df['name'].str.lower()
    # dicts
    gender_mapping_he = names_df[names_df['gender']=='M'].set_index('name_lower')['prob_M'].to_dict()
    gender_mapping_she= names_df[names_df['gender']=='F'].set_index('name_lower')['prob_F'].to_dict()

    # filter only names whose ids are in vocab
    def valid_name(nm):
        return all(tok < tokenizer.vocab_size for tok in names_df.loc[names_df['name_lower']==nm, 'ids'].iloc[0])
    valid_m = {nm: p for nm,p in gender_mapping_he.items() if valid_name(nm)}
    valid_f = {nm: p for nm,p in gender_mapping_she.items() if valid_name(nm)}

    # final lists
    male_names = [(nm, names_df.loc[names_df['name_lower']==nm,'ids'].iloc[0], valid_m[nm]) for nm in valid_m]
    female_names=[(nm, names_df.loc[names_df['name_lower']==nm,'ids'].iloc[0], valid_f[nm]) for nm in valid_f]

    gender_probabilities = {}

    all_texts = []
    for _, record in data.iterrows():
        text = record['text']
        # apply prefix logic as before
        if is_generative and not is_instruction_model:
            text = f'The person, who {text.removeprefix("My friend, [NAME],").removesuffix(".")}, has the first name'
        elif text_prefix:
            text = f'{text_prefix.strip()} {text}'
        elif text_prefix is False:
            text = text.removeprefix('My friend, [NAME],').strip()
        all_texts.append(text)

    for text in all_texts:
        # tokenize context only
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)

        # accumulate weighted probs
        score_M = 0.0
        score_F = 0.0

        # for each male name
        for nm, ids, factor in male_names:
            p = 1.0 # todo this should be the probability of the first token!
            seq = inputs
            for tok in ids:
                with torch.no_grad():
                    logits = model(seq).logits[:, -1, :]
                    probs  = softmax(logits, dim=-1)
                p *= probs[0, tok].item()
                # append token
                seq = torch.cat([seq, torch.tensor([[tok]], device=model.device)], dim=1)
            score_M += p * factor

        # for each female name
        for nm, ids, factor in female_names:
            p = 1.0
            seq = inputs
            for tok in ids:
                with torch.no_grad():
                    logits = model(seq).logits[:, -1, :]
                    probs  = softmax(logits, dim=-1)
                p *= probs[0, tok].item()
                seq = torch.cat([seq, torch.tensor([[tok]], device=model.device)], dim=1)
            score_F += p * factor

        # normalize
        total = score_M + score_F
        P_M = score_M/total if total>0 else 0.0
        P_F = score_F/total if total>0 else 0.0
        gender_probabilities[text] = {'M': P_M, 'F': P_F}

    # metrics: APD, BPI, MPI, FPI, PQ, etc.
    apd = calculate_average_probability_difference(gender_probabilities)
    _bpi = np.mean([(1 - abs(p['M'] - p['F']))*(p['M']+p['F']) for p in gender_probabilities.values()])
    _mpi = np.mean([(1 - p['F'])*p['M'] for p in gender_probabilities.values()])
    _fpi = np.mean([(1 - p['M'])*p['F'] for p in gender_probabilities.values()])
    pq   = calculate_average_prediction_quality(gender_probabilities)
    he_prob = compute_gender_preference_accuracy(gender_probabilities)

    result = {
        'apd': apd,
        'pq': pq,
        '_bpi': _bpi,
        '_mpi': _mpi,
        '_fpi': _fpi,
        'avg_prob_m': np.mean([p['M'] for p in gender_probabilities.values()]),
        'avg_prob_f': np.mean([p['F'] for p in gender_probabilities.values()]),
        'preference_score': abs(np.mean([p['M']-p['F'] for p in gender_probabilities.values()])),
        'he_prob': he_prob,
    }

    if baseline is True:
        result['baseline_change'] = calculate_baseline_change(gender_probabilities, baseline)
        return result, gender_probabilities
    return result


def evaluate_non_gender_mlm(model, tokenizer, max_size=10000):
    from gradiend.setups.gender.en.data import read_geneutral
    df = read_geneutral(max_size=max_size)
    texts = df['text'].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    is_generative = any(name in model.__class__.__name__.lower() for name in ['gpt', 'llama'])
    if is_generative:
        from gradiend.evaluation.mlm import evaluate_clm_perplexity
        result = evaluate_clm_perplexity(model, tokenizer, texts[:1000], verbose=False)
    else:
        from gradiend.evaluation.mlm import evaluate_mlm
        result, stats = evaluate_mlm(model, tokenizer, texts, verbose=False)
    return result


def evaluate_model(model, tokenizer, verbose=True, df_name=None, thorough=True, force=False, cache_folder='', **additional_stats):
    model_hash = hash_model_weights(model)
    model_name = model.name_or_path
    model_id = os.path.basename(model_name)
    if cache_folder and not cache_folder.endswith('/'):
        cache_folder += '/'

    cache_file = f'results/cache/evaluate_model/{model_id}/{cache_folder}{model_hash}.json'
    if not force:
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            return data
        except FileNotFoundError:
            pass

    # Measure the start time of the entire evaluation
    total_start_time = time.time()
    result = {}


    # Evaluate non-gender MLM stats and measure time
    start_time = time.time()
    max_size = 10000 if thorough else 1000
    non_gender_mlm_stats = evaluate_non_gender_mlm(model, tokenizer, max_size=max_size)
    non_gender_mlm_time = time.time() - start_time
    result['mlm'] = non_gender_mlm_stats

    # Evaluate gender bias name stats with prefix and measure time
    start_time = time.time()
    if df_name:
        df_name = f'{df_name.replace(".csv", "")}_friend'
    gender_bias_name_stats = evaluate_gender_bias_name_predictions(model, tokenizer)
    gender_bias_name_time = time.time() - start_time
    result['gender_bias_names'] = gender_bias_name_stats

    # Measure the end time of the entire evaluation
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Calculate relative times
    non_gender_mlm_relative = non_gender_mlm_time / total_time
    gender_bias_name_relative = gender_bias_name_time / total_time

    if verbose:
        # Print nicely formatted times
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print(f"Non-gender MLM evaluation time: {non_gender_mlm_time:.2f} seconds ({non_gender_mlm_relative:.2%} of total time)")
        print(f"Gender bias names evaluation time: {gender_bias_name_time:.2f} seconds ({gender_bias_name_relative:.2%} of total time)")

    accuracy = non_gender_mlm_stats['lms']
    for key in ['bpi', 'mpi', 'fpi']:
        score = gender_bias_name_stats[f'_{key}']
        result[key] = score * accuracy

    if additional_stats:
        result['stats'] = additional_stats

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w+') as f:
        json.dump(result, f, indent=2)

    return result

def get_evaluation_file(path, feature_factors, lrs, thorough=True, top_k=None, part=None, top_k_part=None):
    top_k = f'_top_k_{top_k}' if top_k is not None else ''
    iter_stats = lambda iterable: f'{min(iterable)}_{max(iterable)}_{len(iterable)}{top_k}'
    not_thorough = '_not_thorough' if not thorough else ''
    if part != 'decoder':
        part = '_part_' + part

    if top_k_part != 'decoder':
        top_k_part = f'_top_k_{top_k_part}'

    return f'{path}_{thorough}{top_k}{part}{top_k_part}_evaluation.json', f'{path}_evaluation_{iter_stats(feature_factors)}_{iter_stats(lrs)}{not_thorough}{top_k}{part}{top_k_part}.json'

def convert_results_to_dict(list_results):
    dict_result = {}
    for entry in list_results:
        id = entry['id']
        if isinstance(id, str):
            key = id
        else:
            if 'feature_factor' in id:
                feature_factor = id['feature_factor']
            elif 'gender_factor' in id:
                feature_factor = id['gender_factor']
                print('WARNING: Old id "gender_factor" used, which is deprecated!')
            else:
                raise ValueError(f"Invalid id format: {id}")

            if isinstance(feature_factor, list):
                feature_factor = tuple(feature_factor)

            lr = id['lr']
            key = (feature_factor, lr)

        dict_result[key] = entry
    return dict_result

def convert_results_to_list(dict_results):
    return [{**dict_result, 'id': (key if isinstance(key, str) else {'feature_factor': key[0], 'lr': key[1]})} for key, dict_result in dict_results.items()]


from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

def run_bayesian_optimization(model_with_gradiend,
                              feature_factors,
                              lrs,
                              optimize_metric,
                              accuracy_function,
                              thorough,
                              top_k,
                              part,
                              top_k_part,
                              cache_results,
                              max_evals):
    tokenizer = model_with_gradiend.tokenizer
    latent_dim = model_with_gradiend.gradiend.latent_dim
    evaluated_pairs = set(cache_results.keys())  # base + pairs

    # Define hyperopt search space
    def ff_space():
        if latent_dim == 1:
            return hp.choice("ff0", feature_factors)
        return tuple(hp.choice(f"ff{i}", feature_factors) for i in range(latent_dim))

    space = {
        "feature_factor": ff_space(),
        "lr": hp.choice("lr", lrs)
    }

    def get_metric(result):
        acc = result["mlm"]["accuracy"]
        return accuracy_function(acc) * result["gender_bias_names"][f"_{optimize_metric}"]

    # Hyperopt objective
    def objective(params):
        ff = params["feature_factor"]
        lr = params["lr"]
        key = (ff, lr)

        # CACHE HIT
        if key in cache_results:
            metric = get_metric(cache_results[key])
            return {"loss": -metric, "status": STATUS_OK}

        # CACHE MISS → evaluate
        enhanced = model_with_gradiend.modify_model(
            lr=lr, feature_factor=ff, top_k=top_k, part=part, top_k_part=top_k_part
        )
        result = evaluate_model(
            enhanced, tokenizer, thorough=thorough,
            cache_folder=f"{ff}_{lr}_bayes"
        )

        del enhanced
        torch.cuda.empty_cache()

        cache_results[key] = result  # write to in-memory cache

        metric = get_metric(result)
        return {"loss": -metric, "status": STATUS_OK}

    # Run Bayesian optimization
    trials = Trials()
    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(0)
    )

    # Identify BO-generated pairs
    produced_pairs = set()
    for t in trials.trials:
        vals = t["misc"]["vals"]
        if latent_dim == 1:
            ff = feature_factors[vals["ff0"][0]]
        else:
            ff = tuple(feature_factors[vals[f"ff{i}"][0]] for i in range(latent_dim))
        lr = lrs[vals["lr"][0]]
        produced_pairs.add((ff, lr))

    # Return ONLY pairs not yet evaluated
    missing_pairs = produced_pairs - evaluated_pairs
    return missing_pairs


def evaluate_gradiend(model_with_gradiend,
                      feature_factors=None,
                      lrs=None,
                      thorough=True,
                      top_k=None,
                      accuracy_function=None,
                      part='decoder',
                      top_k_part='decoder',
                      search_strategy='grid', # or 'opt'
                      optimize_metric='bpi',
                      max_opt_evals=1000,
                      ):
    if isinstance(model_with_gradiend, str):
        path_or_model = model_with_gradiend
        model_with_gradiend = ModelWithGradiend.from_pretrained(model_with_gradiend)
    
    path = model_with_gradiend.name_or_path
    accuracy_function = accuracy_function or default_accuracy_function
    metric_keys = ['bpi', 'fpi', 'mpi']

    def apply_accuracy_function(results):
        # recalculate bpi, fpi and mpi with chosen accuracy_function
        raw_results = {k: v.copy() for k, v in results.items() if k not in metric_keys}

        for k, entry in raw_results.items():
            acc = entry['mlm']['lms']
            for key in metric_keys:
                sub_key = f'_{key}'
                value = accuracy_function(acc) * entry['gender_bias_names'][sub_key]
                entry[key] = value

        relevant_results = raw_results.copy()

        for key in metric_keys:
            arg_max = max(raw_results, key=lambda x: raw_results[x][key])
            if arg_max == 'base':
                feature_factor = 0
                lr = 0
            else:
                feature_factor = arg_max[0]
                lr = arg_max[1]
            relevant_results[key] = {
                'value': raw_results[arg_max][key],
                'id': arg_max,
                'feature_factor': feature_factor,
                'lr': lr,
            }
        return relevant_results



    base_model = model_with_gradiend.base_model
    tokenizer = model_with_gradiend.tokenizer
    model_id = os.path.basename(path) if path.startswith('results/models') else path
    base_file, file = get_evaluation_file(f'results/cache/decoder/{model_id}', feature_factors, lrs, thorough=thorough, top_k=top_k, part=part, top_k_part=top_k_part)
    os.makedirs(os.path.dirname(base_file), exist_ok=True)
    os.makedirs(os.path.dirname(file), exist_ok=True)


    try:
        relevant_results = json.load(open(file, 'r'))

        raw_relevant_results = convert_results_to_dict(relevant_results)
        relevant_results = {}
        for k, v in raw_relevant_results.items():
            if 'gender_bias_names' not in v or 'apd' not in v['gender_bias_names'] or not isinstance(v['gender_bias_names']['apd'], dict):
                relevant_results[k] = v
            else:
                print('WARNING')

        # check if complete
    except FileNotFoundError:
        relevant_results = {}
    except Exception as e:
        print(f'Error for {file}')
        raise e

    # Additional check for Bayesian optimization:
    # If search_strategy="opt" and we already have all base + max_evals results,
    # skip BO entirely and return cached results.
    if search_strategy == "opt":
        # count only parameter pairs (excluding 'base' and metric entries)
        cached_pairs = [k for k in relevant_results.keys()
                        if k not in ['base', 'bpi', 'mpi', 'fpi']]

        if len(cached_pairs) >= max_opt_evals:
            # enough BO results in cache → skip optimization
            return apply_accuracy_function(relevant_results)

    if search_strategy == "grid":
        if model_with_gradiend.gradiend.latent_dim == 1:
            pairs = {(feature_factor, lr) for feature_factor in feature_factors for lr in lrs}
        else:
            pairs = set()
            max_pairs = 1000
            for _ in range(max_pairs):
                lr = random.choice(lrs)
                feature_factor = tuple(random.choices(feature_factors, k=model_with_gradiend.gradiend.latent_dim))
                pairs.add((feature_factor, lr))

    elif search_strategy == "opt":
        pairs = run_bayesian_optimization(
            model_with_gradiend=model_with_gradiend,
            feature_factors=feature_factors,
            lrs=lrs,
            optimize_metric=optimize_metric,
            accuracy_function=accuracy_function,
            thorough=thorough,
            top_k=top_k,
            part=part,
            top_k_part=top_k_part,
            cache_results=relevant_results,
            max_evals=max_opt_evals
        )

    expected_results = len(pairs) + 1 + 3 # 1 because of base, 3 because of bpi, mpi, fpi




    if len(relevant_results) == expected_results:
        return apply_accuracy_function(relevant_results)
    try:
        all_results = json.load(open(base_file, 'r'))
        raw_all_results = convert_results_to_dict(all_results)

        all_results = {}
        for k, v in raw_all_results.items():
            if 'gender_bias_names' not in v or 'apd' not in v['gender_bias_names'] or not isinstance(v['gender_bias_names']['apd'], dict):
                all_results[k] = v

        # copy relevant results into relevant_results
        for pair in pairs:
            if pair in all_results:
                relevant_results[pair] = all_results[pair]

        if 'base' in all_results:
            relevant_results['base'] = all_results['base']
        elif ('b', 'a') in all_results:
            # todo deprecated because of earlier error
            relevant_results['base'] = all_results[('b', 'a')]

        if len(relevant_results) == expected_results:
            with open(file, 'w+') as f:
                json.dump(convert_results_to_list(relevant_results), f, indent=2)
            return apply_accuracy_function(relevant_results)

    except FileNotFoundError:
        all_results = {}

    if 'base' in relevant_results:
        print("Skipping base model as it is already evaluated")
    else:
        base_results = evaluate_model(base_model, tokenizer, force=True)
        all_results['base'] = base_results
        relevant_results['base'] = base_results


    for feature_factor, lr in tqdm(pairs, desc=f"Evaluate GRADIEND {path_or_model}"):
        id = {'feature_factor': feature_factor, 'lr': lr}
        id_key = (feature_factor, lr)
        if id_key in relevant_results:
            #print(f"Skipping {id} as it is already evaluated")
            continue

        enhanced_bert = model_with_gradiend.modify_model(lr=lr, feature_factor=feature_factor, top_k=top_k, part=part, top_k_part=top_k_part)
        if top_k is None:
            df_name = f'results/cache/models/{model_with_gradiend.name}_lr_{lr}_gf_{feature_factor}.csv'
        else:
            df_name = f'results/cache/models/{model_with_gradiend.name}/{top_k}/lr_{lr}_gf_{feature_factor}.csv'

        enhanced_bert_results = evaluate_model(enhanced_bert, tokenizer, df_name=df_name, thorough=thorough, cache_folder=f'{feature_factor}_{lr}_v2') # force=True?
        all_results[id_key] = enhanced_bert_results
        relevant_results[id_key] = enhanced_bert_results

        with open(base_file, 'w+') as f:
            json.dump(convert_results_to_list(all_results), f, indent=2)

        # free memory
        del enhanced_bert
        torch.cuda.empty_cache()


    list_results = convert_results_to_list(relevant_results)
    with open(file, 'w+') as f:
        json.dump(list_results, f, indent=2)
    return apply_accuracy_function(convert_results_to_dict(list_results))

def plot_gradiend_model_selection(data,
                                  model_name,
                                  feature_factors=None,
                                  lrs=None,
                                  metrics=None,
                                  highlight_metrics=None,
                                  friend=True,
                                  split='total',
                                  thorough=True,
                                  highlight='best',
                                  small=True,
                                  square=True,
                                  plot_numbers=True,
                                  mirror_lr_axis=False,
                                  single_plots=False,
                                  overview_plot=False,
                                  linear_plot=True,
                                  horizontal_plot=False,
                                  **kwargs,
                                  ):
    metrics = metrics or ['avg_prob_m', 'avg_prob_f', 'accuracy', 'bpi', 'fpi', 'mpi']
    highlight_metrics = highlight_metrics or ['bpi', 'fpi', 'mpi']
    model_name = model_name.removeprefix('results/models/')
    baseline = None
    evaluations = []

    gf_x_axis = False

    if isinstance(data, list):
        data = convert_results_to_dict(data)

    for id, entry in data.items():
        if id == 'base':
            baseline = entry
        else:
            evaluations.append(entry)

    if baseline is None:
        print(json.dumps(data, indent=2))
        raise ValueError('No baseline found in data!')

    if feature_factors is None or lrs is None:
        if feature_factors is None:
            feature_factors = []
        if lrs is None:
            lrs = []


        for id, entry in data.items():
            if id == 'base':
                continue

            feature_factor = entry['id']['feature_factor']
            lr = entry['id']['lr']
            if feature_factor not in feature_factors:
                feature_factors.append(feature_factor)

            if lr not in lrs:
                lrs.append(lr)

        # sort the lists
        feature_factors = list(sorted(feature_factors))
        lrs = list(sorted(lrs))



    def get_metric(x, metric):
        bias = 'gender_bias_names' if friend else 'gender_bias'
        if '->' in metric:
            parts = metric.split('->')
            for part in parts:
                if part in x:
                    x = x[part]
                elif parts.index(part) == len(parts) - 1:
                    return 0.0
                else:
                    raise ValueError(f"Metric part '{part}' not found in data (available keys: {list(x.keys())}, full metric was '{metric}')")
        elif metric in x:
            return x[metric]
        elif metric == 'accuracy':
            x = x['mlm']['lms']
        elif metric in {'gender_preference_accuracy', 'overall_accuracy', 'average_absolute_difference'}:
            x = x[''] # todo what? Deprecated??
        elif metric == 'avg_prob_m + avg_prob_f':
            if isinstance(x[bias]['avg_prob_m'], dict):
                x = {k: xx + x[bias]['avg_prob_f'][k] for k, xx in x[bias]['avg_prob_m'].items()}
            else:
                x = x[bias]['avg_prob_m'] + x[bias]['avg_prob_f']
        elif metric in {'bpi', 'fpi', 'mpi'}:
            x = normalization(x[metric])
        else:
            x = x[bias][metric]

        if isinstance(x, float):
            return x
        return x[split]

        # Prepare the subplots

    baseline_values = {metric: get_metric(baseline, metric) * 100 for metric in metrics}

    metrics_labels = {
        'apd': 'APD = Average Probability Difference = Sum(|P(M) - P(F)|)',
        'preference_score': 'Preference Score = |mean(P(M)) - mean(P(F))|',
        'avg_prob_f': 'Mean Probability of female token = mean(P(F))',
        'avg_prob_m': 'Mean Probability of male token = mean(P(M))',
        'he_prob': 'Mean for M has higher probability than F (P(M) > P(F))',
        'accuracy': 'MLM accuracy on non-gender text',
    }

    metric_labels_latex = {
        'apd': 'APD',
        'bpi': 'BPI',
        'fpi': 'FPI',
        'mpi': 'MPI',
        'avg_prob_f': r'$\mathbb{P}(F)$',
        'avg_prob_m': r'$\mathbb{P}(M)$',
        'avg_prob_m + avg_prob_f': r'$\mathbb{P}(F\cup M)$',
        'accuracy': r'$\text{LMS}_\text{Dec}$',
        'white': r'$\mathbb{P}(White)$',
        'black': r'$\mathbb{P}(Black)$',
        'asian': r'$\mathbb{P}(Asian)$',
        'christian': r'$\mathbb{P}(Christian)$',
        'muslim': r'$\mathbb{P}(Muslim)$',
        'jewish': r'$\mathbb{P}(Jewish)$',
        'lms': r'$\text{LMS}_\text{Dec}$',
        'race_black_asian': r'BalancedBS',
        'race_white_black': r'BalancedBS',
        'race_white_asian': r'BalancedBS',
        'religion_christian_jewish': r'BalancedBS',
        'religion_christian_muslim': r'BalancedBS',
        'religion_muslim_jewish': r'BalancedBS',
    }


    metric_dfs = {}
    extreme_values = {}
    heatmap_data_dfs = {}
    masks = {}
    lr_name = r'Learning Rate $\alpha$'
    gf_name = 'Feature Factor $h$'

    for metric in metrics:
        df = pd.DataFrame([{
            'feature_factor': e['id']['feature_factor'],
            'lr': e['id']['lr'],
            'value': get_metric(e, metric)
        } for e in evaluations if isinstance(e, dict) and isinstance(e['id'], dict)])
        metric_dfs[metric] = df

        if mirror_lr_axis:
            df['lr'] = df['lr'].apply(lambda x: -x)

        # rename column feature_factor to "Gender Factor"
        df[gf_name] = df['feature_factor']
        df[lr_name] = df['lr']

        if gf_x_axis:
            heatmap_data = df.pivot(index=lr_name, columns=gf_name, values='value')[::-1]
        else:
            heatmap_data = df.pivot(index=gf_name, columns=lr_name, values='value')[::-1]


        heatmap_data_dfs[metric] = heatmap_data
        baseline_value = baseline_values[metric]

        if metric in {'preference_score', 'apd'}:
            mask = heatmap_data < baseline_value
            extreme_value = heatmap_data.min().min()
        else:
            mask = heatmap_data > baseline_value
            extreme_value = heatmap_data.max().max()
        extreme_values[metric] = extreme_value
        masks[metric] = mask


    if gf_x_axis:
        gf_data = heatmap_data.columns
        lr_data = heatmap_data.index
    else:
        gf_data = heatmap_data.index
        lr_data = heatmap_data.columns
    gf_tick_labels = [f'{factor:.1f}' for factor in gf_data]
    lr_tick_labels = [f'{lr:.0e}'.replace('e-0', 'e-') for lr in lr_data]

    metric_colors = {
        'bpi': '#FF0000',
        'mpi': '#FFA500',
        'fpi': '#FF00FF',
    }
    tolerance = 0.01

    def get_evaluation_file(path, feature_factors, lrs, thorough=True):
        iter_stats = lambda iterable: f'{min(iterable)}_{max(iterable)}_{len(iterable)}'
        not_thorough = '_not_thorough' if not thorough else ''
        return f'results/models/{model_name}_evaluation{not_thorough}.json', f'results/models/{model_name}_evaluation_{iter_stats(feature_factors)}_{iter_stats(lrs)}{not_thorough}.json'

    _, image_file = get_evaluation_file(f'img/{model_name}', feature_factors, lrs, thorough=thorough)
    base_image_file = image_file.replace('.json', f'{"_small" if small else ""}{"_square" if square else ""}.pdf')

    if overview_plot:

        n_metrics = len(metrics)
        if n_metrics == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            axes = [axes]
        elif n_metrics == 2 or n_metrics == 3:
            fig, axes = plt.subplots(1, n_metrics, figsize=(10 * n_metrics, 8))
        elif n_metrics == 4:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            axes = axes.flatten()
        elif n_metrics in [5, 6]:
            fig, axes = plt.subplots(2, 3, figsize=(30, 16))
            axes = axes.flatten()
        elif n_metrics in [7, 8]:
            fig, axes = plt.subplots(2, 4, figsize=(40, 16))
            axes = axes.flatten()
        else:
            raise ValueError(f"Too many metrics to plot: {n_metrics} ({metrics}")

        for ax, metric in zip(axes, metrics):
            metric_label = metric_labels_latex.get(metric, metric)
            heatmap_data = heatmap_data_dfs[metric]
            baseline_value = baseline_values[metric]
            mask = masks[metric]
            kwargs = {'xticklabels' if gf_x_axis else 'yticklabels': gf_tick_labels}
            sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': ''}, annot=True, fmt='.2f',
                        annot_kws={"size": 9 if small else 10}, ax=ax,
                        **kwargs)

            for y in range(heatmap_data.shape[0]):
                for x in range(heatmap_data.shape[1]):
                    if highlight == 'best':
                        for metric in highlight_metrics:
                            metric_color = metric_colors.get(metric, '#FF0000')
                            heatmap_data_metric = heatmap_data_dfs[metric]
                            extreme_value = extreme_values[metric]
                            if heatmap_data_metric.iloc[y, x] == extreme_value:
                                ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=metric_color, linewidth=3))
                    else:
                        extreme_value = extreme_values[metric]
                        if abs(heatmap_data.iloc[y, x] - extreme_value) <= tolerance:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', linewidth=3))
                        elif mask.iloc[y, x]:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red'))

            ax.set_title(f"{metric_label}", fontsize=15)
            # Add a custom text for the baseline value with background color
            ax.text(1.0,
                    1.0,
                    f"(Baseline: {baseline_value:.4f})",
                    transform=ax.transAxes,  # Set relative to the axis
                    fontsize=10,  # Font size
                    ha='right',  # Horizontal alignment
                    va='center',  # Vertical alignment
                    bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))  # Background color

        fig.suptitle(model_name, y=1, fontsize=25)
        plt.tight_layout(rect=[0, 0, 0.98, 1])

        os.makedirs(os.path.dirname(base_image_file), exist_ok=True)
        plt.savefig(base_image_file, bbox_inches='tight')
        plt.show()

    if single_plots:
        for idx, metric in enumerate(metrics):
            version = base_image_file.removesuffix(".pdf").removeprefix('results/models/').removeprefix(model_name).removeprefix('_')
            heatmap_data = heatmap_data_dfs[metric]
            baseline_value = baseline_values[metric]
            mask = masks[metric]

            # Create a new figure for each subplot
            if small:
                if square:
                    fig_single, ax_single = plt.subplots(figsize=(4, 2.9))
                else:
                    fig_single, ax_single = plt.subplots(figsize=(4, 2.0))
            else:
                fig_single, ax_single = plt.subplots(figsize=(8, 3.3))

            # Plot the heatmap again for each subplot
            cbar_kws = {}
            kwargs = {
                'xticklabels': lr_tick_labels,
                'yticklabels': gf_tick_labels,
            }
            heatmap_data_percent = heatmap_data * 100

            max_val = heatmap_data_percent.max().max()
            annot_size = 9 if small else 8
            annot_size = 7
            if max_val < 1:
                fmt = ".2f"  # all values tiny → keep 2 decimals
                annot_size = 4
            elif max_val < 10:
                fmt = ".1f"  # moderate values → one decimal
                annot_size = 5
            else:
                fmt = ".0f"  # large values → integers only

            sns.heatmap(heatmap_data_percent, cmap="YlGnBu", cbar_kws=cbar_kws, annot=plot_numbers, fmt=fmt,
                        annot_kws={"size": annot_size, "va": "center_baseline"}, ax=ax_single, **kwargs)
            ls = (10 if square else 8) if small else 10
            ax_single.tick_params(axis='x', labelsize=ls)
            ax_single.tick_params(axis='y', labelsize=ls)
            cbar = ax_single.collections[0].colorbar
            cbar.ax.tick_params(labelsize=ls)

            for y in range(heatmap_data.shape[0]):
                for x in range(heatmap_data.shape[1]):
                    if highlight == 'best':
                        for metric_ in highlight_metrics:
                            metric_color = metric_colors.get(metric_, '#FF0000')
                            heatmap_data_metric = heatmap_data_dfs[metric_]
                            extreme_value = extreme_values[metric_]
                            if heatmap_data_metric.iloc[y, x] == extreme_value:
                                ax_single.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=metric_color, linewidth=2))
                    else:
                        extreme_value = extreme_values[metric]
                        if abs(heatmap_data.iloc[y, x] - extreme_value) <= tolerance:
                            ax_single.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', linewidth=3))
                        elif mask.iloc[y, x]:
                            ax_single.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red'))

            # Determine the color based on the baseline_value and the cmap YlGnBu
            cmap = plt.get_cmap("YlGnBu")  # Get the colormap

            # Normalize baseline_value to a range between 0 and 1
            norm = matplotlib.colors.Normalize(vmin=heatmap_data_percent.min().min(), vmax=heatmap_data_percent.max().max())

            # Map the baseline_value to the corresponding color in the colormap
            baseline_color = cmap(norm(baseline_value))

            # Function to compute luminance and decide text color
            def get_text_color(facecolor):
                # Convert color to RGB
                rgb = matplotlib.colors.to_rgb(facecolor)
                # Calculate luminance using relative luminance formula
                luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

                # If luminance is high (bright background), return black text, else return white text
                return 'black' if luminance > 0.5 else 'white'

            text_color = get_text_color(baseline_color)

            if small:
                kwargs = {'x': 1.21, 'y': -0.26}
            else:
                kwargs = {'x': 0.992, 'y': -0.24}
            plt.text(
                s=f'Base Model: {baseline_value:.1f}',  # Text with formatted value
                color=text_color,  # Text color
                fontsize=10 if small else 9,  # Font size
                ha='right',  # Align text to the right
                va='bottom',  # Align text to the bottom
                bbox={  # Background box styling
                    'facecolor': baseline_color,
                    'pad': 2,
                    'edgecolor': 'none'
                },
                transform=plt.gca().transAxes,  # Use axes coordinates
                **kwargs
            )

            #plt.suptitle(f'{model_name} - {metric}')
            fs = (13 if square else 11) if small else 12
            if gf_x_axis:
                ax_single.set_xlabel(gf_name, fontsize=fs)
                ax_single.set_ylabel(lr_name, fontsize=fs)
            else:
                ax_single.set_xlabel(lr_name, fontsize=fs)
                ax_single.set_ylabel(gf_name, fontsize=fs)

            if small:
                ax_single.yaxis.set_label_coords(-0.145, 0.5)

            # Save the individual subplot
            metric_id = metrics[idx].replace(' ', '')
            image_file = f'img/decoder/{model_name}/{version}/subplot_{metric_id}.pdf'
            os.makedirs(os.path.dirname(image_file), exist_ok=True)
            fig_single.savefig(image_file, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig_single)  # Close to prevent memory issues

    if linear_plot:
        # prepare subplot grid
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), squeeze=False)
        # 1 row, n_metrics columns

        for idx, metric in enumerate(metrics):
            ax = axes[0, idx]  # select subplot axis

            version = base_image_file.removesuffix(".pdf").removeprefix('results/models/').removeprefix(
                model_name).removeprefix('_')
            heatmap_data = heatmap_data_dfs[metric]
            baseline_value = baseline_values[metric]
            mask = masks[metric]

            # --- heatmap plotting ---
            cbar_kws = {}
            kwargs = {
                'xticklabels': lr_tick_labels,
                'yticklabels': gf_tick_labels,
            }
            heatmap_data_percent = heatmap_data * 100

            max_val = heatmap_data_percent.max().max()
            annot_size = 7
            if max_val < 1:
                fmt = ".2f"
                annot_size = 4
            elif max_val < 10:
                fmt = ".1f"
                annot_size = 5
            else:
                fmt = ".0f"

            sns.heatmap(
                heatmap_data_percent,
                cmap="YlGnBu",
                cbar_kws=cbar_kws,
                annot=False,
                fmt=fmt,
                #annot_kws={"size": annot_size, "va": "center_baseline"},
                ax=ax,
                **kwargs
            )

            ls = 14
            ax.tick_params(axis='x', labelsize=ls)
            ax.tick_params(axis='y', labelsize=ls)

            # add latex title
            metric_label = metric_labels_latex.get(metric.split('->')[-1], metric)
            ax.set_title(f"{metric_label}", fontsize=15 if n_metrics == 4 else 23)

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=ls)

            # highlight rectangles (kept same as your logic)
            for y in range(heatmap_data.shape[0]):
                for x in range(heatmap_data.shape[1]):
                    if highlight == 'best':
                        for metric_ in highlight_metrics:
                            metric_color = metric_colors.get(metric, '#FF0000')
                            heatmap_data_metric = heatmap_data_dfs[metric_]
                            extreme_value = extreme_values[metric_]
                            if heatmap_data_metric.iloc[y, x] == extreme_value:
                                ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=metric_color, linewidth=2))
                    else:
                        extreme_value = extreme_values[metric]
                        if abs(heatmap_data.iloc[y, x] - extreme_value) <= tolerance:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', linewidth=3))
                        elif mask.iloc[y, x]:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red'))

            # baseline color logic (unchanged)
            cmap = plt.get_cmap("YlGnBu")
            norm = matplotlib.colors.Normalize(vmin=heatmap_data_percent.min().min(), vmax=heatmap_data_percent.max().max())
            baseline_color = cmap(norm(baseline_value))

            def get_text_color(facecolor):
                rgb = matplotlib.colors.to_rgb(facecolor)
                luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                return 'black' if luminance > 0.5 else 'white'

            text_color = get_text_color(baseline_color)

            if False:
                baseline_value_fmt = f"{baseline_value:.1f}"
                if baseline_value_fmt == '0.0':
                    baseline_value_fmt = f"{baseline_value:.2f}"
                    if baseline_value_fmt == '0.00':
                        baseline_value_fmt = f"{baseline_value:.3f}"
                        if baseline_value_fmt == '0.000':
                            baseline_value_fmt = f"{baseline_value:.4f}"
                text_y = -0.4 if n_metrics == 4 else -0.3
                ax.text(
                    0.9, text_y, f'Base: {baseline_value_fmt}',
                    color=text_color,
                    fontsize=13,
                    ha='left',
                    va='bottom',
                    bbox={'facecolor': baseline_color, 'pad': 2, 'edgecolor': 'none'},
                    transform=ax.transAxes
                )

            # labels
            fs = 16 if n_metrics == 4 else 20
            fs_y = fs - 1 if n_metrics == 4 else fs
            if gf_x_axis:
                ax.set_xlabel(gf_name, fontsize=fs)
                ax.set_ylabel(lr_name, fontsize=fs_y)
            else:
                ax.set_xlabel(lr_name, fontsize=fs)
                ax.set_ylabel(gf_name, fontsize=fs_y)

        # tighten layout
        plt.tight_layout()

        # save combined figure
        image_file = f'img/decoder/{model_name}.pdf'
        os.makedirs(os.path.dirname(image_file), exist_ok=True)
        fig.savefig(image_file, dpi=300, bbox_inches='tight')
        print(f'Saved linear overview plot to {image_file}')
        plt.show()
        plt.close(fig)

    if horizontal_plot:
        # prepare subplot grid
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(5, 4 * n_metrics), squeeze=False)
        # 1 row, n_metrics columns

        for idx, metric in enumerate(metrics):
            ax = axes[idx, 0]  # select subplot axis

            version = base_image_file.removesuffix(".pdf").removeprefix('results/models/').removeprefix(
                model_name).removeprefix('_')
            heatmap_data = heatmap_data_dfs[metric]
            baseline_value = baseline_values[metric]
            mask = masks[metric]

            # --- heatmap plotting ---
            cbar_kws = {}
            kwargs = {
                'xticklabels': lr_tick_labels,
                'yticklabels': gf_tick_labels,
            }
            heatmap_data_percent = heatmap_data * 100

            max_val = heatmap_data_percent.max().max()
            annot_size = 7
            if max_val < 1:
                fmt = ".2f"
                annot_size = 4
            elif max_val < 10:
                fmt = ".1f"
                annot_size = 5
            else:
                fmt = ".0f"

            sns.heatmap(
                heatmap_data_percent,
                cmap="YlGnBu",
                cbar_kws=cbar_kws,
                annot=False,
                fmt=fmt,
                #annot_kws={"size": annot_size, "va": "center_baseline"},
                ax=ax,
                **kwargs
            )

            ls = 14
            ax.tick_params(axis='x', labelsize=ls)
            ax.tick_params(axis='y', labelsize=ls)

            # add latex title
            metric_label = metric_labels_latex.get(metric.split('->')[-1], metric)
            ax.set_title(f"{metric_label}", fontsize=25 if n_metrics == 4 else 23)

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=ls)

            # highlight rectangles (kept same as your logic)
            for y in range(heatmap_data.shape[0]):
                for x in range(heatmap_data.shape[1]):
                    if highlight == 'best':
                        for metric_ in highlight_metrics:
                            metric_color = metric_colors.get(metric, '#FF0000')
                            heatmap_data_metric = heatmap_data_dfs[metric_]
                            extreme_value = extreme_values[metric_]
                            if heatmap_data_metric.iloc[y, x] == extreme_value:
                                ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=metric_color, linewidth=2))
                    else:
                        extreme_value = extreme_values[metric]
                        if abs(heatmap_data.iloc[y, x] - extreme_value) <= tolerance:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', linewidth=3))
                        elif mask.iloc[y, x]:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red'))

            # baseline color logic (unchanged)
            cmap = plt.get_cmap("YlGnBu")
            norm = matplotlib.colors.Normalize(vmin=heatmap_data_percent.min().min(), vmax=heatmap_data_percent.max().max())
            baseline_color = cmap(norm(baseline_value))

            def get_text_color(facecolor):
                rgb = matplotlib.colors.to_rgb(facecolor)
                luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                return 'black' if luminance > 0.5 else 'white'

            # labels
            fs = 21 if n_metrics == 4 else 20
            fs_y = fs - 1 if n_metrics == 4 else fs
            if gf_x_axis:
                ax.set_xlabel(gf_name, fontsize=fs)
                ax.set_ylabel(lr_name, fontsize=fs_y)
            else:
                ax.set_xlabel(lr_name, fontsize=fs)
                ax.set_ylabel(gf_name, fontsize=fs_y)

        # tighten layout
        plt.tight_layout()

        # save combined figure
        image_file = f'img/decoder/{model_name}.pdf'
        os.makedirs(os.path.dirname(image_file), exist_ok=True)
        fig.savefig(image_file, dpi=300, bbox_inches='tight')
        print(f'Saved linear overview plot to {image_file}')
        plt.show()
        plt.close(fig)

    return data

# currently not used function!
def top_neuron_evaluation(model, bin_size=100, lr=None, feature_factor=None, key='bpi', thorough=True, plot=True):
    if lr is None or feature_factor is None:
        print(f'Determining best best choice for lr and gender factor based on default_evaluation and key {key}')
        evaluation = default_evaluation(model, large=True, plot=False)
        best = evaluation[key]
        print(f'Best choice for {key} is {best}')
        if lr is None:
            lr = best['lr']
        if feature_factor is None:
            feature_factor = best['feature_factor']


    bert_with_ae = ModelWithGradiend.from_pretrained(model)
    base_model = bert_with_ae.base_model
    tokenizer = bert_with_ae.tokenizer

    def get_evaluation_file(path, feature_factor, lr, bin_size):
        not_thorough = '_not_thorough' if not thorough else ''
        return f'{path}_top_k_evaluation_gf_{feature_factor}_lr_{lr:.1e}{not_thorough}.json', f'{path}_top_k_evaluation_bs_{bin_size}_gf_{feature_factor}_lr_{lr:.1e}{not_thorough}.json'

    base_file, file = get_evaluation_file(model + f'/../cache/decoder/' + os.path.basename(model), feature_factor, lr, bin_size)
    os.makedirs(os.path.dirname(base_file), exist_ok=True)
    os.makedirs(os.path.dirname(file), exist_ok=True)

    # n is number of elements in the GRADIEND model
    n = bert_with_ae.gradiend.input_dim
    borders = [0, 40000]
    borders.append(n-30000)
    borders.append(n)

    expected_results = len(borders) # + 3 because of bpi, mpi, fpi
    keys = ['bpi', 'mpi', 'fpi']

    def do_plot():
        if plot:
            # plot the results
            fig, ax = plt.subplots()
            x = list(borders)
            for key in ['bpi']:
                y = [relevant_results[str(k)][key] for k in x]
                ax.plot([str(xx) for xx in x], y, label=key)
            ax.set_xlabel('Top K')
            # rotate x tick labels by 45 degrees
            plt.xticks(rotation=45)
            plt.grid()
            plt.legend()
            plt.show()

    try:
        relevant_results = json.load(open(file, 'r'))

        # check if complete
        if len(relevant_results) == expected_results:
            do_plot()
            return relevant_results
    except FileNotFoundError:
        relevant_results = {}
    except Exception as e:
        print(f'Error for {file}')
        raise e

    try:
        #raise FileNotFoundError()
        all_results = json.load(open(base_file, 'r'))

        # copy relevant results into relevant_results
        for border in borders:
            if border in all_results:
                relevant_results[str(border)] = all_results[border]

        if len(relevant_results) == expected_results:
            with open(file, 'w+') as f:
                json.dump(relevant_results, f, indent=2)
            return relevant_results

    except FileNotFoundError:
        all_results = {}

    for border in tqdm(borders, desc="Evaluate BERT With AE"):
        id = str(border)

        if id in relevant_results:
            print(f"Skipping {id} as it is already evaluated")
            continue

        raw_enhanced_bert_results = default_evaluation(model, plot=False, large=False, top_k=border)
        enhanced_bert_results = {key: raw_enhanced_bert_results[key]['value'] for key in keys}
        print(f"Results for {id}: {enhanced_bert_results}")
        print('Genderfactor', raw_enhanced_bert_results[key]['feature_factor'])
        print('Learning Rate', raw_enhanced_bert_results[key]['lr'])

        all_results[id] = enhanced_bert_results
        relevant_results[id] = enhanced_bert_results

        with open(base_file, 'w+') as f:
            json.dump(all_results, f, indent=2)

    raw_relevant_results = {k: v for k, v in relevant_results.items() if k not in keys}
    for key in keys:
        top_k = max(raw_relevant_results, key=lambda x: raw_relevant_results[x][key])

        relevant_results[key] = {
            'value': relevant_results[top_k][key],
            'top_k': top_k,
        }

    with open(file, 'w+') as f:
        json.dump(relevant_results, f, indent=2)

    pprint(relevant_results)

    do_plot()

    return relevant_results

default_evaluation_feature_factors = [-10, -2] + [-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0] + [2, 10]
default_evaluation_lrs = [-0.5, -1e-1, -5e-2, -1e-2, -5e-3, -1e-3, -5e-4, -1e-4,  1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5] # -5e-5, -1e-5, 1e-5, 5e-5,
def default_evaluation(model, large=True, plot=True, top_k=None, accuracy_function=None, part='decoder', top_k_part='decoder', search_strategy='grid', **kwargs):
    if large:
        feature_factors = default_evaluation_feature_factors
        lrs = default_evaluation_lrs
        #lrs = [-5e-2, -1e-2, -5e-3, -1e-3, -5e-4, -1e-4, -5e-5, -1e-5, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    else:
        feature_factors = [-1, 0, 1]
        lrs = [1e-2, 1e-3, 1e-4, 1e-5]

    data = evaluate_gradiend(model, feature_factors, lrs, thorough=large, top_k=top_k, accuracy_function=accuracy_function, part=part, top_k_part=top_k_part, search_strategy=search_strategy)

    if plot:
        model = os.path.basename(model.removesuffix('model_with_gradiend'))
        plot_gradiend_model_selection(data, model, feature_factors=feature_factors, lrs=lrs, thorough=large, single_plots=True, **kwargs)

    return data


def test_aggregation_functions(model):
    accuracy_conversion_functions = {
        '^1': lambda x: x,
        '^2': np.square,
        #'^3': lambda x: np.power(x, 3),
        #'^4': lambda x: np.power(x, 3),
        '^5': lambda x: np.power(x, 3),
        '^10': lambda x: np.power(x, 10),
        #'2^': lambda x: np.power(2, x),
    }

    model_id = os.path.basename(model) if model.startswith('results/models') else model
    base_file, _ = get_evaluation_file(f'results/cache/decoder/{model_id}', default_evaluation_feature_factors, default_evaluation_lrs)

    def bpi_fpi_mpi(stats, accuracy_conversion_function):
        acc = stats['mlm']['lms']
        converted_acc = accuracy_conversion_function(acc)

        bpi = converted_acc * stats['gender_bias_names']['_bpi']
        fpi = converted_acc * stats['gender_bias_names']['_fpi']
        mpi = converted_acc * stats['gender_bias_names']['_mpi']

        return bpi, fpi, mpi


    results = json.load(open(base_file, 'r'))
    new_results = []
    for function_name, accuracy_conversion_function in accuracy_conversion_functions.items():
        for entry in results:
            bpi, fpi, mpi = bpi_fpi_mpi(entry, accuracy_conversion_function)
            new_entry = entry.copy()
            new_entry['bpi'] = bpi
            new_entry['fpi'] = fpi
            new_entry['mpi'] = mpi
            new_results.append(new_entry)

        output_file = f"{base_file.removesuffix('.json')}_{function_name}.json"
        with open(output_file, 'w+') as f:
            json.dump(new_results, f, indent=2)

        plot_gradiend_model_selection(new_results, f'{model_id}_{function_name}')



def evaluate_models():
    model1 = 'results/changed_models/bert-base-uncased/female'
    model2 = 'results/changed_models/bert-base-uncased/male'
    model5 = 'results/changed_models/bert-base-uncased-unbiased'
    model3 = 'results/changed_models/bert-large-cased/female'
    model4 = 'results/changed_models/bert-large-cased/male'
    model6 = 'results/changed_models/bert-large-cased-unbiased'
    models = [
        model1, model2, 'bert-base-uncased', 'bert-large-cased', model3, model4,model5, model6
    ]
    results = {}
    metrics = []
    for model_id in models:
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = AutoTokenizerForLM.from_pretrained(model_id)
        result = evaluate_model(model, tokenizer, verbose=True)
        print(model_id)
        #print(json.dumps(result, indent=2))
        results[model_id] = result
        apd = result['gender_bias_names']['apd']['total']
        P_M = result['gender_bias_names']['avg_prob_m']['total']
        P_F = result['gender_bias_names']['avg_prob_f']['total']
        metrics.append((model_id, apd, P_M, P_F))

    print(json.dumps(metrics, indent=2))

def evaluate_gender_prediction_metrics(results_df):
    # Initialize a dictionary to store metrics
    metrics = {}

    # Get all unique splits from the DataFrame
    splits = results_df['split'].unique()

    # Calculate accuracy for each split
    for split in splits:
        split_df = results_df[results_df['split'] == split]
        accuracy = split_df['correct'].mean()  # Mean of correct predictions (True = 1, False = 0)
        male_accuracy = split_df[split_df['true_gender'] == 'M']['correct'].mean()
        female_accuracy = split_df[split_df['true_gender'] == 'F']['correct'].mean()

        metrics[split] = {
            'accuracy': accuracy,
            'male_accuracy': male_accuracy,
            'female_accuracy': female_accuracy,
            #'total_samples': len(split_df),
            #'correct_predictions': split_df['correct'].sum(),
            'male_prob_mean': split_df['male_prob'].mean(),
            'female_prob_mean': split_df['female_prob'].mean(),
        }

    # Calculate overall accuracy
    overall_accuracy = results_df['correct'].mean()
    male_accuracy = results_df[results_df['true_gender'] == 'M']['correct'].mean()
    female_accuracy = results_df[results_df['true_gender'] == 'F']['correct'].mean()

    metrics['total'] = {
        'accuracy': overall_accuracy,
        'male_accuracy': male_accuracy,
        'female_accuracy': female_accuracy,
        #'total_samples': len(results_df),
        #'correct_predictions': results_df['correct'].sum(),
        'male_prob_mean': results_df['male_prob'].mean(),
        'female_prob_mean': results_df['female_prob'].mean()
    }

    df = pd.DataFrame(metrics).T
    # add the index as column "split"
    df['split'] = df.index
    return df

def evaluate_gender_prediction(model_name, tokenizer=None, target=None, target_words=('he', 'she')):
    assert target in {None, 'M', 'F'}
    tokenizer = tokenizer or model_name
    model = AutoModelForLM.from_pretrained(model_name)
    tokenizer = AutoTokenizerForLM.from_pretrained(tokenizer)
    return evaluate_gender_prediction_by_model(model, tokenizer, target=target, target_words=target_words)

def evaluate_gender_prediction_by_model(model, tokenizer, target=None, target_words=('he', 'she'), verbose=True):
    model.eval()
    is_generative = tokenizer.cls_token is None

    if is_generative:
        tokenizer.pad_token = tokenizer.eos_token

    from gradiend.setups.gender.en import read_namexact
    names = read_namexact()

    # Convert male and female tokens to IDs once
    target_word_1, target_word_2 = target_words
    target_word_F, target_word_M = target_words
    if target_word_F.lower() in {'he', 'man'}:
        target_word_M, target_word_F = target_word_F, target_word_M

    target_words_M = [target_word_M]
    target_words_F = [target_word_F]

    if is_generative:
        # make the first character upper
        target_words_M.append(target_word_M.capitalize())
        target_words_F.append(target_word_F.capitalize())

    if hasattr(tokenizer, 'tokenizer'):
        raw_tokenizer = tokenizer.tokenizer
    else:
        raw_tokenizer = tokenizer

    male_token_ids = [raw_tokenizer(f'{space}{target_word_M}', add_special_tokens=False)['input_ids'][0] for target_word_M in target_words_M for space in ['', ' ']]
    female_token_ids = [raw_tokenizer(f'{space}{target_word_F}', add_special_tokens=False)['input_ids'][0] for target_word_F in target_words_F for space in ['', ' ']]

    male_token_ids = list(set(male_token_ids))
    female_token_ids = list(set(female_token_ids))

    results = []

    for split, split_names in names.groupby('split'):
        split_data = []
        genders = []
        names_list = []

        # Collect all the text and gender info for this split
        for _, (name, gender) in split_names[['name', 'gender']].iterrows():
            if is_generative:
                text = f'Classify the following name as either "{target_word_1}" or "{target_word_2}": {name}  \nAnswer:'
                text = f'Classify the following name as either "man" or "woman". Respond with only one word — either "{target_word_1}" or "{target_word_2}". No explanation. \nName: {name}\nAnswer (respond with only one word - either {target_word_1} or {target_word_2}): "'
            else:
                text = f'Is {name} a "{target_word_1}" or a "{target_word_2}"? {name} is a {tokenizer.mask_token}.'

            split_data.append(text)
            genders.append(gender)
            names_list.append(name)

        # Tokenize the entire batch of text at once
        tokenized_text = tokenizer(split_data, return_tensors="pt", padding=True, truncation=True)
        tokenized_text = {k: v.to(model.device) for k, v in tokenized_text.items()}

        with torch.no_grad():
            # Get model outputs (logits) for the batch
            outputs = model(**tokenized_text)
            logits = outputs.logits

        is_instruction_model = 'instruct' in model.name_or_path.lower()
        if is_generative:
            # get the index of the last token before the padding starts
            last_token_index = (tokenized_text['input_ids'] != tokenizer.pad_token_id).sum(dim=-1)
            if is_instruction_model:
                last_token_index += 1
            else:
                last_token_index -= 1
            relevant_logits = logits[range(len(last_token_index)), last_token_index]
        else:
            # Get the index of the [MASK] token
            mask_token_index = (tokenized_text['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            # Extract logits for the [MASK] token positions
            relevant_logits = logits[mask_token_index]

        # Calculate probabilities for male and female tokens using softmax over all predictions
        probabilities = torch.softmax(relevant_logits, dim=-1)

        # Extract male and female probabilities for each name in the batch
        male_probs = probabilities[:, male_token_ids]
        female_probs = probabilities[:, female_token_ids]

        top_k = 5
        # Get top k predictions for each masked token position
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)

        # Decode the top k token indices to their string representations
        top_k_tokens = [[raw_tokenizer.decode([i]) for i in indices] for indices in top_k_indices]

        # Process the results for each item in the batch
        for i, (name, gender, male_probs, female_probs, top_tokens, top_probs) in enumerate(zip(names_list, genders, male_probs, female_probs, top_k_tokens, top_k_probs)):
            male_prob = male_probs.sum()
            female_prob = female_probs.sum()

            # Predicted gender based on the highest probability
            predicted_gender = "M" if male_prob > female_prob else "F"
            if target is None:
                correct_prediction = (predicted_gender == gender)
            elif target == 'M':
                correct_prediction = (predicted_gender == 'M')
            elif target == 'F':
                correct_prediction = (predicted_gender == 'F')
            else:
                raise ValueError('Invalid target', target)

            # Collect the result for this name
            results.append({
                'name': name,
                'true_gender': gender,
                'predicted_gender': predicted_gender,
                'correct': correct_prediction,
                'male_prob': male_prob.item(),
                'female_prob': female_prob.item(),
                'split': split,
                'top_tokens': top_tokens,
            })

            if verbose:
                print(f"Name: {name}")
                print(f"Top {top_k} Predictions: {', '.join([f'{token} ({prob:.2f})' for token, prob in zip(top_tokens, top_probs)])}")
                print(f"True Gender: {gender}, Predicted Gender: {predicted_gender}, Male Prob: {male_prob:.2f}, Female Prob: {female_prob:.2f}")
                print()
                verbose = False

    # Return the results as a DataFrame
    df = pd.DataFrame(results)

    metrics = evaluate_gender_prediction_metrics(df)

    return metrics, df


def evaluate_gender_prediction_for_models(*models, target_words=('man', 'woman')):
    suffix = '_'.join(target_words)
    for base_model in models:
        base_model_id = base_model.split('/')[-1]
        models = [
            base_model,
            f'results/changed_models/{base_model_id}-N',
            f'results/changed_models/{base_model_id}-M',
            f'results/changed_models/{base_model_id}-F',
        ]

        results = []
        for model in models:
            print(f'Evaluating {model} with target words {target_words}')
            df, df_raw = evaluate_gender_prediction(model, tokenizer=base_model, target_words=target_words)
            df['model'] = model
            model_id = os.path.basename(model)
            output_predictions = f'results/gender_prediction/{model_id}_{suffix}_predictions.csv'
            df_raw.to_csv(output_predictions, index=False)
            results.append(df)

        df = pd.concat(results)
        output = f'results/gender_prediction/{base_model_id}_{suffix}.csv'
        os.makedirs(os.path.dirname(output), exist_ok=True)
        df.to_csv(output, index=False)



def evaluate_all_gender_predictions():
    evaluate_gender_prediction_for_models(
        'bert-base-cased',
        'bert-large-cased',
        'distilbert-base-cased',
        'roberta-large',
        #'gpt2',
        #'meta-llama/Llama-3.2-3B',
        #'meta-llama/Llama-3.2-3B-Instruct',
        target_words=('man', 'woman'),
    )

    evaluate_gender_prediction_for_models(
        'bert-base-cased',
        'bert-large-cased',
        'distilbert-base-cased',
        'roberta-large',
        #'gpt2',
        #'meta-llama/Llama-3.2-3B',
        #'meta-llama/Llama-3.2-3B-Instruct',
        target_words=('woman', 'man'),
    )


if __name__ == '__main__':
    evaluate_all_gender_predictions()