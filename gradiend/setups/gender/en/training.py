import itertools
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

from gradiend.data import read_gerneutral
from gradiend.data.util import get_file_name, json_dumps
from gradiend.evaluation.analyze_decoder import default_evaluation
from gradiend.evaluation.analyze_encoder import get_model_metrics
from gradiend.model import ModelWithGradiend
from gradiend.setups import Setup
from gradiend.setups.emotion import plot_encoded_by_class
from gradiend.setups.gender.en import GenderEnSetup, create_training_dataset, get_gender_words, read_namexact, \
    read_genter
from gradiend.training.gradiend_training import train_for_configs, train as train_gradiend
from gradiend.util import get_files_and_folders_with_prefix


class MultiDimGenderEnSetup(Setup):

    def __init__(self, n_features=2):
        super().__init__('gender-en', n_features=n_features)

    def create_training_data(self, *args, **kwargs):
        return create_training_dataset(*args, **kwargs)

    def evaluate_(self, model_with_gradiend, eval_data, eval_batch_size=32, config=None, training_stats=None):
        # one hot encode the labels
        #if isinstance(eval_data['labels'][0], int):
        #    num_classes = max(eval_data['labels']) + 1
        #    eval_data['labels'] = np.eye(num_classes)[eval_data['labels']]

        result = super().evaluate(model_with_gradiend, eval_data, eval_batch_size=eval_batch_size)
        score = result['score']
        encoded = result['encoded']
        encoded_by_class = result['encoded_by_class']
        mean_by_class = result['mean_by_class']

        output_name = f'training_{str(model_with_gradiend.gradiend.encoder[1])}.pdf'
        if config and 'output' in config:
            base_output = config['output']
            global_step = training_stats.get('global_step', None)
            output = f'{base_output}/{global_step}_{output_name}'
        else:
            output = f'img/{output_name}'
        plot_encoded_by_class(encoded_by_class, mean_by_class=mean_by_class, title=f"Score {score}", output=output)

        return result

    def analyze_decoder(self, *args, search_strategy='opt', plot=None, **kwargs):
        return default_evaluation(search_strategy=search_strategy, plot=False, *args, **kwargs)

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

        tokenizer = model_with_gradiend.tokenizer
        mask_token = tokenizer.mask_token
        is_generative = model_with_gradiend.is_generative
        is_llama = 'llama' in tokenizer.name_or_path.lower()

        modified_cache = []

        source = model_with_gradiend.source

        male_names = itertools.cycle(names_df[names_df['gender'] == 'M'].iterrows())
        female_names = itertools.cycle(names_df[names_df['gender'] == 'F'].iterrows())

        filled_texts = []

        def process_entry(row):
            if is_generative:
                masked = row['masked'].split('[PRONOUN]')[0]
            else:
                masked = row['masked'].replace('[PRONOUN]', mask_token)
            encoded_values = []
            genders = []
            names = []
            labels = []

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
                    encoded = model_with_gradiend.gradiend.encoder(grads)
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
                        filled_text = filled_text.strip()

                    encoded = model_with_gradiend.encode(filled_text, label=masked_label, top_k=top_k, top_k_part=top_k_part)

                if hasattr(encoded, 'tolist'):
                    encoded = encoded.tolist()

                if len(encoded) == 1:
                    encoded = encoded[0]

                encoded_values.append(encoded)
                genders.append(entry['gender'])
                names.append(name)
                labels.append([label] * row['pronoun_count'])

            results = pd.DataFrame({
                'text': masked,
                'name': names,
                'state': genders,
                'encoded': encoded_values,
                'labels': labels,
                'type': 'gender masked',
            })

            results['state_value'] = results['state'].map({'M': 0, 'F': 1, 'N': 0.5})
            results = results.sort_values(by='state')

            return results

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

            result = pd.DataFrame({
                'text': texts,
                'name': None,
                'state': None,
                'encoded': encoded_values,
                'token_distance': None,
                'type': 'gerneutral',
                'label': labels,
            })
            results.append(result)

        texts = []
        encoded_values = []
        labels = []
        torch.manual_seed(42)
        for text in tqdm(filled_texts, desc='GENTER data without gender words masked'):
            encoded, masked_text, label = model_with_gradiend.mask_and_encode(text, ignore_tokens=gender_tokens,
                                                                              return_masked_text=True, top_k=top_k, top_k_part=top_k_part)
            if encoded is None:
                continue

            texts.append(text)
            encoded_values.append(encoded)
            labels.append(label)


        result = pd.DataFrame({
            'text': texts,
            'name': None,
            'state': None,
            'encoded': encoded_values,
            'token_distance': None,
            'type': 'no gender masked',
            'label': labels,
        })
        results.append(result)

        total_results = pd.concat(results)

        total_results['label'] = total_results['label'].apply(json_dumps)

        total_results.to_csv(output, index=False)

        return total_results

    def get_model_metrics(self, output, split='test'):
        return get_model_metrics(output, split=split, v=3)



def default_training(configs):
    setup = GenderEnSetup()
    train_for_configs(setup, configs, n=1)

def dual_training(configs, version=None, activation='tanh'):
    setup = MultiDimGenderEnSetup(n_features=2)
    for id, config in configs.items():
        config['activation'] = activation
        config['delete_models'] = True
    train_for_configs(setup, configs, version=version, n=3)



def multi_dim_training(configs, version=None, activation='tanh', n_features=2):
    setup = MultiDimGenderEnSetup(n_features=n_features)
    for id, config in configs.items():
        config['activation'] = activation
        config['delete_models'] = True
    train_for_configs(setup, configs, version=version, n=2)



if __name__ == '__main__':
    model_configs = {
        'bert-base-cased': dict(),
        'bert-large-cased': dict(eval_max_size=0.5, eval_batch_size=4),
        'distilbert-base-cased': dict(),
        'roberta-large': dict(eval_max_size=0.5, eval_batch_size=4),
        'gpt2': dict(),
        'meta-llama/Llama-3.2-3B-Instruct': dict(batch_size=32, eval_max_size=0.05, eval_batch_size=1, epochs=1, torch_dtype=torch.bfloat16, lr=1e-4),
        'meta-llama/Llama-3.2-3B': dict(batch_size=32, eval_max_size=0.05, eval_batch_size=1, epochs=1, torch_dtype=torch.bfloat16, lr=1e-4, n_evaluation=250),
    }

    setup = GenderEnSetup()

    models = []
    for base_model, model_config in model_configs.items():
        model = train_gradiend(setup, base_model, model_config, n=3, version='', clear_cache=False, force=False)
        models.append(model)

    for model in models:
        setup.select(model)
