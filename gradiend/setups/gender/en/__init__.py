import itertools
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset

from gradiend.data import read_gerneutral
from gradiend.data.util import get_file_name, json_dumps
from gradiend.evaluation.analyze_decoder import default_evaluation
from gradiend.evaluation.analyze_encoder import get_model_metrics
from gradiend.model import ModelWithGradiend
from gradiend.setups import Setup
from gradiend.setups.gender.en.data import read_namexact, read_genter, read_geneutral, get_gender_words
from gradiend.setups.gender.en.util import read_default_predictions, write_default_predictions
from gradiend.util import get_files_and_folders_with_prefix, evaluate_he_she, token_distance, z_score


class TrainingDataset(Dataset):

    """Initializes the GRADIEND training dataset.

    Args:
    - data (pd.DataFrame): The GENTER template training data.
    - names (pd.DataFrame): The names dataset.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for tokenization.
    - batch_size (int): The batch size used for the training. Each data entry will be paired with batch_size random names.
    - neutral_data (pd.DataFrame): The neutral data used for the training between the gender-related maskings. If None, no neutral data is used (default).
     - neutral_data_prop (float): The proportion of neutral data used in the training. If 0, no neutral data is used (default).
     - gender_words (list): A list of gender specific words that are excluded for masking of neutral_data.
    - max_token_length (int): The maximum token length of the input (48 are sufficient for most of the GENTER data)

    In each batch, a random gender is chosen and only this names of this gender are used for the entire batch (with
    different random texts). At the end of the training, each entry of data will be paired exactly batch_size many times with a name.
    This behavior is made deterministic and is automatically applied when using __getitem__.
     """
    def __init__(self,
                 data,
                 names,
                 tokenizer,
                 batch_size=None,
                 neutral_data=None,
                 neutral_data_prop=0.0,
                 gender_words=None,
                 max_token_length=48,
                 is_generative=False
                 ):
        self.data = data
        self.names = names
        self.names = self.names[self.names['gender'].isin(['M', 'F'])]
        self.tokenizer = tokenizer
        if is_generative:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.mask_token = self.tokenizer.mask_token
        self.batch_size = None if not batch_size else batch_size
        self.neutral_data = neutral_data if neutral_data_prop > 0 else None
        self.neutral_data_prop = neutral_data_prop
        self.is_generative = is_generative
        self.is_instruct_model = 'instruct' in self.tokenizer.name_or_path.lower()
        if self.is_instruct_model:
            self.gender_pronoun_tokens = {(gender, upper): self.tokenizer.encode(pronoun[0].upper() + pronoun[1:] if upper else pronoun, add_special_tokens=False)[0] for gender, pronoun in [('M', ' he'), ('F', ' she')] for upper in [True, False]}
        else:
            self.gender_pronoun_tokens = {(gender, upper): self.tokenizer.encode(pronoun[0].upper() + pronoun[1:] if upper else pronoun, add_special_tokens=False)[0] for gender, pronoun in [('M', 'he'), ('F', 'she')] for upper in [True, False]}
        self._analyze_name_tokens()

        gender_words = gender_words or []
        self.gender_token_ids = list(set(token for word in gender_words for token in self.tokenizer(word, add_special_tokens=False)['input_ids']))

        deterministic_random = random.Random(42)
        self.index_sampling = deterministic_random.sample(range(self.len_without_batch()), self.len_without_batch())
        self.full_index_sampling = deterministic_random.sample(range(self.__len__()), len(self))
        self.max_token_length = max_token_length

    """Returns the number of raw entries in the dataset. If the neutral data is used,
    the length is calculated as if the neutral data was part of the main data."""
    def len_without_batch(self):
        n = len(self.data)
        if self.neutral_data is not None:
            n = int(n / (1 - self.neutral_data_prop))
        return n

    """Returns the number of entries in the dataset, considering the batch size."""
    def __len__(self):
        n = self.len_without_batch()

        if self.batch_size == None:
            return n
        else:
            return n * self.batch_size

    def get_data(self, index):
        is_gender_data = True
        data = self.data
        offset = index % self.batch_size if self.batch_size is not None else 0
        idx = index // self.batch_size if self.batch_size is not None else index
        random_idx = self.index_sampling[idx]

        if self.neutral_data is not None and random_idx >= len(self.data):
            is_gender_data = False
            data = self.neutral_data
            random_index = random_idx - len(self.data)
        else:
            random_index = self.full_index_sampling[random_idx]

        final_index = (random_index + offset) % len(data)

        entry = data.iloc[final_index]
        return entry, is_gender_data, idx, offset


    def _analyze_name_tokens(self):
        self.gender_name_tokens = {}

        for gender, df in self.names.groupby('gender'):
            # Dictionary to store the mapping
            token_dict = defaultdict(dict)
            names = df['name'].unique()

            # Tokenize each name and store in the dictionary
            for name in names:
                # Tokenize the name
                tokens = self.tokenizer.encode(name, add_special_tokens=False)
                token_count = len(tokens)

                # Add the name to the dictionary based on the number of tokens
                token_dict[token_count][name] = tokens
            self.gender_name_tokens[gender] = token_dict

        # sanitize, i.e. make sure that the token counts are available for each gender
        # Find common token counts across all genders
        common_token_counts = set(self.gender_name_tokens[next(iter(self.gender_name_tokens))].keys())
        for gender in self.gender_name_tokens:
            common_token_counts.intersection_update(self.gender_name_tokens[gender].keys())

        # Filter the token dictionaries to only include common token counts
        for gender in self.gender_name_tokens:
            self.gender_name_tokens[gender] = {token_count: names
                                          for token_count, names in self.gender_name_tokens[gender].items()
                                          if token_count in common_token_counts}

        self.token_count_distribution = {count: min(len(self.gender_name_tokens[gender][count]) for gender in self.gender_name_tokens)
                                         for count in self.gender_name_tokens['M']}

    def get_random_name(self, gender, seed):
        names = [(k, v) for names in self.gender_name_tokens[gender].values() for k, v in names.items()]
        random_name = random.Random(seed).choice(names)
        return {'name': random_name[0], 'tokens': random_name[1]}

    def tokenize(self, text):
        item = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_token_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in item.items()}
        return item

    def mask_tokens(self, inputs, mask_probability=0.15):
        """
        Mask tokens in the input tensor for MLM, excluding certain tokens.

        Args:
        - inputs (torch.Tensor): The tokenized inputs.
        - mask_probability (float): The probability of masking a token.

        Returns:
        - masked_inputs (torch.Tensor): The inputs with some tokens masked.
        - labels (torch.Tensor): The labels for the MLM task.
        """
        labels = inputs.clone()

        # Create a mask to exclude certain token IDs from masking
        exclude_mask = (inputs.unsqueeze(-1) == torch.Tensor(self.gender_token_ids)).any(dim=-1)

        # Create a random mask for the inputs with the given probability
        random_mask = torch.rand(inputs.shape, dtype=torch.float) < mask_probability

        padding_mask = inputs == self.tokenizer.pad_token_id

        # Apply the exclusion mask to ensure excluded tokens are not masked
        mask = random_mask & ~exclude_mask & ~padding_mask

        # Mask the tokens in the inputs
        masked_inputs = inputs.clone()
        masked_inputs[mask] = self.mask_token_id

        # For labels, we only want to compute loss on masked tokens
        labels[~mask] = -100

        return masked_inputs, labels


    def __getitem__(self, index):
        entry, is_gender_data, batch_size_index, batch_size_offset = self.get_data(index)

        if is_gender_data:
            masked = entry['masked']

            gender = random.Random(batch_size_index).choice(['M', 'F'])
            inverse_gender = 'F' if gender == 'M' else 'M'

            random_name = self.get_random_name(gender, batch_size_offset)

            def fill(text, name):
                filled_text = text.replace('[NAME]', name)
                if self.is_generative:
                    return filled_text.split('[PRONOUN]')[0]
                else:
                    return filled_text.replace('[PRONOUN]', self.mask_token)
            gender_text = fill(masked, random_name['name'])

            item = self.tokenize(gender_text)
            gender_labels = item['input_ids'].clone()
            inv_gender_labels = gender_labels.clone()

            sentence_delimiter = {'.', '!', '?'}
            if self.is_generative:
                upper = any([gender_text.strip().endswith(x) for x in sentence_delimiter])
                mask_token_mask = torch.zeros_like(gender_labels, dtype=torch.bool)
                last_idx = (gender_labels != self.tokenizer.pad_token_id).nonzero()[-1].item()
                mask_token_mask[last_idx] = True
            else:
                mask_token_mask = gender_labels == self.mask_token_id

                # check if the pronoun start with an uppercase letter since it appears at the beginning of the sentence
                gender_text_no_white_spaces = gender_text.replace(' ', '')
                mask_index = gender_text_no_white_spaces.index(self.mask_token)
                upper = mask_index == 0 or mask_index > 2 and gender_text_no_white_spaces[mask_index - 1] in sentence_delimiter and gender_text_no_white_spaces[mask_index - 2] != '.' # avoid "..."
            gender_labels[~mask_token_mask] = -100 # only compute loss on masked tokens
            gender_labels[mask_token_mask] = self.gender_pronoun_tokens[(gender, upper)] # set masked tokens to the
            inv_gender_labels[~mask_token_mask] = -100 # only compute loss on masked tokens
            inv_gender_labels[mask_token_mask] = self.gender_pronoun_tokens[(inverse_gender, upper)] # set masked tokens to the

            inv_item = item.copy()
            item['labels'] = gender_labels
            inv_item['labels'] = inv_gender_labels

            label = int(gender == 'F') # 1 -> F, 0 -> M
            text = gender_text
            factual_target = 'he' if label == 'M' else 'she'
            counterfactual_target = 'she' if label == 'M' else 'he'
        else:
            # this dataset is used for general knowledge data (not gender-specific)
            text = entry['text']

            item = self.tokenize(text)
            masked_input, labels = self.mask_tokens(item['input_ids'])
            item['input_ids'] = masked_input
            item['labels'] = labels
            inv_item = item
            label = ''
            gender = ''
            factual_target = ''
            counterfactual_target = ''

        return {
            True: item,
            False: inv_item,
            'text': text,
            'label': label,
            'binary_label': label,
            'metadata': gender,
            'factual_target': factual_target,
            'counterfactual_target': counterfactual_target,
        }

def create_training_dataset(tokenizer,
                            max_size=None,
                            batch_size=None,
                            neutral_data=False,
                            split='train',
                            neutral_data_prop=0.5,
                            is_generative=False,
                            ):
    # Load datasets
    names = read_namexact(split=split)
    dataset = read_genter(split=split)
    if max_size:
        dataset = dataset.iloc[range(max_size)]

    neutral_data = read_geneutral(max_size=len(dataset), split=split) if neutral_data else None

    gender_words = get_gender_words()

    # Create custom dataset
    max_token_length = 128 if 'llama' in tokenizer.name_or_path.lower() else 48
    return TrainingDataset(dataset,
                           names,
                           tokenizer,
                           batch_size=batch_size,
                           neutral_data=neutral_data,
                           gender_words=gender_words,
                           neutral_data_prop=neutral_data_prop,
                           is_generative=is_generative,
                           max_token_length=max_token_length
                           )


def plot_model_results(encoded_values, title='', y='z_score'):
    if isinstance(encoded_values, str):
        results = read_encoded_values(encoded_values)
        if not title:
            title = encoded_values.removesuffix('.csv')
    else:
        results = encoded_values

    plot_results = results.copy()
    plot_results['plot_state'] = plot_results['type']
    if y == 'encoded':
        plot_results.loc[plot_results['type'] == 'gender masked', 'plot_state'] = plot_results['state']
    else:
        plot_results = plot_results[plot_results['type'] == 'gender masked'].reset_index(drop=True)
        plot_results['plot_state'] = plot_results['state']

#    plot_results = results[results['type'] == 'gender masked'].sort_values(by='state').reset_index(drop=True)
    sns.boxplot(x='plot_state', y=y, data=plot_results)
    plt.grid()
    plt.title(title)
    file = f'img/z_score/{title}_{y}.png'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    plt.savefig(file)
    plt.show()

class GenderEnSetup(Setup):

    def __init__(self):
        super().__init__('gender-en')
        self.metric_keys = ['bpi', 'fpi', 'mpi']
        self.version_map = {
            'bpi': 'N',
            'fpi': 'F',
            'mpi': 'M'
        }

    def get_model_metrics(self, output, split='test'):
        return get_model_metrics(output, split=split, v=3)

    def create_training_data(self, *args, **kwargs):
        return create_training_dataset(*args, **kwargs)

    def analyze_decoder(self, *args, **kwargs):
        return default_evaluation(*args, **kwargs)

    def analyze_models(self,
                       *models,
                       max_size=None,
                       force=False,
                       split='test',
                       prefix=None,
                       best_score=None,
                       top_k=None,
                       top_k_part=None,
                       return_output_path=False,
                       ):
        if prefix:
            # find all models in the folder with the suffix
            best_score = '_best' if best_score else ''
            models = list(models) + get_files_and_folders_with_prefix(prefix, only_folder=True, suffix=best_score)
        print(f'Analyze {len(models)} Models:', models)

        names_df = read_namexact(split=split)
        df = read_genter(split=split)
        df_no_gender = read_gerneutral(max_size=min(10000, len(df) * 2))

        if max_size:
            df = df.head(max_size)
            df_no_gender = df_no_gender.head(max_size)

        dfs = {}
        for model in models:

            output = get_file_name(model, max_size=max_size, file_format='csv', split=split, v=3, top_k=top_k, top_k_part=top_k_part)

            if force or not os.path.isfile(output):
                model_with_ae = ModelWithGradiend.from_pretrained(model)
                analyze_df = self.analyze_model(model_with_ae, df, names_df, output=output, df_no_gender=df_no_gender, top_k=top_k, top_k_part=top_k_part)
                print(f'Done with Model {model}')

            else:
                print(f'Skipping Model {model} as output file {output} already exists!')
                analyze_df = pd.read_csv(output)

            if len(models) == 1:
                if return_output_path:
                    return output
                return analyze_df
            if return_output_path:
                dfs[model] = output
            else:
                dfs[model] = analyze_df
        return dfs

    def analyze_model(self, 
                      model_with_gradiend, 
                      genter_df, 
                      names_df, 
                      output, 
                      df_no_gender=None, 
                      include_B=False,
                      plot=False,
                      top_k=None,
                      top_k_part=None,
                      ):
        if not include_B and 'B' in names_df['gender'].unique():
            if not 'genders' in names_df:
                raise ValueError('The names_df must contain the key genders if include_B is False')
            names_df = names_df[names_df['genders'] != 'B']

        model = model_with_gradiend.base_model
        tokenizer = model_with_gradiend.tokenizer
        mask_token = tokenizer.mask_token
        is_generative = model_with_gradiend.is_generative
        is_llama = 'llama' in tokenizer.name_or_path.lower()

        cache_default_predictions_dict = read_default_predictions(model)

        modified_cache = []

        def get_default_prediction(masked_text):
            if masked_text in cache_default_predictions_dict:
                return cache_default_predictions_dict[masked_text]

            # calculate the predictions
            predictions = evaluate_he_she(model, tokenizer, masked_text)
            cache_default_predictions_dict[masked_text] = predictions
            if not modified_cache:
                modified_cache.append(True)
            return predictions

        source = model_with_gradiend.source

        male_names = itertools.cycle(names_df[names_df['gender'] == 'M'].iterrows())
        female_names = itertools.cycle(names_df[names_df['gender'] == 'F'].iterrows())

        filled_texts = []

        def process_entry(row, plot=False):
            if is_generative:
                masked = row['masked'].split('[PRONOUN]')[0]
            else:
                masked = row['masked'].replace('[PRONOUN]', mask_token)
            encoded_values = []
            genders = []
            names = []
            token_distances = []
            labels = []
            default_predictions = {k: [] for k in ['he', 'she', 'most_likely_token', 'label']}

            masked_token_distance = token_distance(tokenizer, masked, '[NAME]', mask_token)

            # todo make the encoding batched!

            inputs = []
            for i, entry in [next(female_names), next(male_names)]:
                name = entry['name']
                filled_text = masked.replace('[NAME]', name)
                filled_texts.append(filled_text)
                label = 'he' if entry['gender'] == 'M' else 'she'

                if source == 'diff':
                    label_factual = 'he' if label == 'M' else 'she'
                    label_counter_factual = 'she' if label == 'M' else 'he'
                    inputs_factual = model_with_gradiend.create_inputs(filled_text, label_factual)
                    grads_factual = model_with_gradiend.forward_pass_create_gradients(inputs_factual, return_dict=False)
                    inputs_counter_factual = model_with_gradiend.create_inputs(filled_text, label_counter_factual)
                    grads_counter_factual = model_with_gradiend.forward_pass_create_gradients(inputs_counter_factual,
                                                                                              return_dict=False)
                    grads = grads_factual - grads_counter_factual
                    inputs.append(grads)
                    encoded = model_with_gradiend.gradiend.encoder(grads).item()
                else:
                    if source == 'factual':
                        masked_label = label
                    elif source == 'counterfactual':
                        masked_label = 'he' if label == 'she' else 'she'
                    else:
                        raise ValueError(f'Unknown source: {source}')
                    inputs.append((filled_text, masked_label))
                    if is_llama:
                        masked_label = f' {masked_label}' # todo lammainstr
                        filled_text = filled_text.strip() # todo lammainstr

                    encoded = model_with_gradiend.encode(filled_text, label=masked_label, top_k=top_k, top_k_part=top_k_part)

                if hasattr(encoded, 'tolist'):
                    encoded = encoded.tolist()

                if len(encoded) == 1:
                    encoded = encoded[0]

                encoded_values.append(encoded)
                genders.append(entry['gender'])
                names.append(name)
                token_distances.append(masked_token_distance)
                labels.append([label] * row['pronoun_count'])

                default_prediction = get_default_prediction(filled_text)
                default_prediction['label'] = label
                for key, value in default_prediction.items():
                    default_predictions[key].append(value)

            results = pd.DataFrame({
                'text': masked,
                'name': names,
                'state': genders,
                'encoded': encoded_values,
                'token_distance': token_distances,
                'labels': labels,
                'type': 'gender masked',
                **default_predictions,
            })

            results['state_value'] = results['state'].map({'M': 0, 'F': 1, 'B': 0.5})
            results['z_score'] = z_score(results['encoded'])
            results = results.sort_values(by='state')

            if plot:
                plt.title(row['masked'])
                sns.boxplot(x='state', y='z_score', data=results)
                plt.show()

            return results

            # todo correlate with names that might be used for the other gender!!!

        # Enable the progress_apply method for DataFrames
        tqdm.pandas(desc="Analyze with GENTER Test Data")
        genter_df['genter'] = genter_df.progress_apply(process_entry, axis=1)
        results = genter_df['genter'].tolist()

        gender_tokens = get_gender_words(tokenizer=tokenizer)
        torch.manual_seed(42)
        if df_no_gender is not None:
            if len(df_no_gender) > 10000:
                df_no_gender = df_no_gender.head(10000).reset_index(drop=True)
            texts = []
            encoded_values = []
            labels = []
            default_predictions = {k: [] for k in ['he', 'she', 'most_likely_token', 'label']}

            for i, row in tqdm(df_no_gender.iterrows(), desc='No gender data', total=len(df_no_gender)):
                text = row['text']
                encoded, masked_text, label = model_with_gradiend.mask_and_encode(text, ignore_tokens=gender_tokens,
                                                                                  return_masked_text=True,
                                                                                  single_mask=True, top_k=top_k, top_k_part=top_k_part)
                if encoded is None:
                    continue
                texts.append(masked_text)
                encoded_values.append(encoded)
                labels.append(label)

                default_prediction = get_default_prediction(masked_text)
                default_prediction['label'] = label
                for key, value in default_prediction.items():
                    default_predictions[key].append(value)

            result = pd.DataFrame({
                'text': texts,
                'name': None,
                'state': None,
                'encoded': encoded_values,
                'token_distance': None,
                'type': 'gerneutral',
                **default_predictions,
            })
            results.append(result)

        texts = []
        encoded_values = []
        labels = []
        default_predictions = {k: [] for k in ['he', 'she', 'most_likely_token', 'label']}
        torch.manual_seed(42)
        for text in tqdm(filled_texts, desc='GENTER data without gender words masked'):
            encoded, masked_text, label = model_with_gradiend.mask_and_encode(text, ignore_tokens=gender_tokens,
                                                                              return_masked_text=True, top_k=top_k, top_k_part=top_k_part)
            if encoded is None:
                continue

            texts.append(text)
            encoded_values.append(encoded)
            labels.append(label)

            default_prediction = get_default_prediction(masked_text)
            default_prediction['label'] = label
            for key, value in default_prediction.items():
                default_predictions[key].append(value)

        result = pd.DataFrame({
            'text': texts,
            'name': None,
            'state': None,
            'encoded': encoded_values,
            'token_distance': None,
            'type': 'no gender masked',
            **default_predictions,
        })
        results.append(result)

        if modified_cache:
            write_default_predictions(cache_default_predictions_dict, model)

        total_results = pd.concat(results)

        total_results['he'] = total_results['he'].apply(json_dumps)
        total_results['she'] = total_results['she'].apply(json_dumps)
        total_results['label'] = total_results['label'].apply(json_dumps)
        total_results['most_likely_token'] = total_results['most_likely_token'].apply(json_dumps)

        total_results.to_csv(output, index=False)

        if plot:
            # plot results
            plot_model_results(total_results, title=output.removesuffix('.csv'))

            plot_results = total_results[total_results['type'] == 'gender masked'].sort_values(by='state').reset_index(
                drop=True)
            sns.boxplot(x='state', y='encoded', data=plot_results)
            plt.title(model_with_gradiend.name_or_path)
            plt.show()

            cor = np.nanmean(
                [text_df[['encoded', 'state_value']].corr(method='pearson')['encoded']['state_value'] for text, text_df
                 in plot_results.groupby('text')])
            print('Correlation', cor)

            # now without B
            plot_results_MF = plot_results[plot_results['state'] != 'B']
            cor = np.nanmean(
                [text_df[['encoded', 'state_value']].corr(method='pearson')['encoded']['state_value'] for text, text_df
                 in plot_results_MF.groupby('text')])
            print('Correlation MF', cor)

        return total_results

    def analyze_models_with_other_data(
            self,
            model_with_gradiend,
            input_csv="data/other_gender_data_balanced.csv",
            top_k=None,
            top_k_part=None,
            plot=False
        ):
        import pandas as pd
        import torch
        from tqdm import tqdm
        tqdm.pandas()

        output = f"{model_with_gradiend.name_or_path}/other_gender_analysis.csv"
        if False and os.path.isfile(output):
            return pd.read_csv(output)


        df = pd.read_csv(input_csv)
        tokenizer = model_with_gradiend.tokenizer
        mask_token = tokenizer.mask_token
        is_llama = 'llama' in tokenizer.name_or_path.lower()

        def process_row(row):
            masked_text = row["masked_text"]
            true_label = row["token"]  # actual token used for mask

            # replace placeholder with model mask token
            if model_with_gradiend.is_generative:
                masked_input = masked_text.split('[MASK]')[0]
            else:
                masked_input = masked_text.replace("[MASK]", mask_token)
            if is_llama:
                masked_input = masked_input.strip()

            try:
                encoded = model_with_gradiend.encode(
                    masked_input,
                    label=true_label,
                    top_k=top_k,
                    top_k_part=top_k_part
                )
            except Exception:
                return {
                    'text': row['text'],
                    'masked_text': masked_text,
                "token": row["token"],
                "group": row["group"],
                "label": row["label"],  # numeric +1/-1
                "encoded": None,
                }

            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if isinstance(encoded, list) and len(encoded) == 1:
                encoded = encoded[0]

            return {
                "text": row["text"],
                "masked_text": masked_text,
                "token": row["token"],
                "group": row["group"],
                "label": row["label"],  # numeric +1/-1
                "encoded": encoded
            }

        tqdm.pandas(desc="Analyzing balanced gender data")
        results = df.progress_apply(process_row, axis=1).tolist()
        results_df = pd.DataFrame(results)
        results_df.to_csv(output, index=False)

        if plot:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.boxplot(x="group", y="pred", data=results_df)
            plt.title(model_with_gradiend.name_or_path)
            plt.show()

        return results_df

    def get_other_model_metrics(self, analysis):
        if isinstance(analysis, str):
            analysis = pd.read_csv(analysis)

        def safe_pearson(x, y):
            if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
                return float("nan")
            return pearsonr(x, y)[0]

        preds = analysis["encoded"].values
        labels = analysis["label"].values

        metrics = {"overall_pearson": safe_pearson(preds, labels)}

        # per token
        token_corrs = {}
        for token, group in analysis.groupby("token"):
            token_corrs[token] = safe_pearson(group["encoded"].values, group["label"].values)
        metrics["per_token"] = token_corrs

        # per group
        group_corrs = {}
        for grp, group in analysis.groupby("group"):
            group_corrs[grp] = safe_pearson(group["encoded"].values, group["label"].values)
        metrics["per_group"] = group_corrs

        return metrics


