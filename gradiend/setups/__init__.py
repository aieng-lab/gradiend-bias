import json
import os
import random
import time
from collections import defaultdict

import torch
from sklearn.linear_model import Ridge
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from gradiend.data.util import get_file_name
from gradiend.evaluation.analyze_encoder import get_model_metrics
from gradiend.model import ModelWithGradiend, AutoModelForLM
from gradiend.training import ModelFeatureTrainingDataset
import gc

from itertools import combinations

from gradiend.util import convert_tuple_keys_to_strings

py_print = print

class Setup:

    def __init__(self, id, n_features=1):
        self.id = id
        self.n_features = n_features

        self.metric_keys = []
        self.version_map = {}
        self.non_neutral_terms = []
        self.larger_is_better = True
        self.init_gradiends = []

    @property
    def pretty_id(self):
        return self.id

    @property
    def create_training_data(self):
        raise NotImplementedError("Subclasses should implement this method to create training data.")


    def create_eval_data(self, *args, **kwargs):
        return create_eval_dataset(self, *args, **kwargs)


    def _post_training(self, base_output, step=None, **kwargs):
        pass

    def evaluate_old(self, model_with_gradiend, eval_data, eval_batch_size=32, **kwargs):
        start = time.time()
        grads = eval_data['gradients']
        labels = eval_data['labels']
        label_dim = len(labels[0]) if hasattr(labels[0], '__len__') else 1
        assert label_dim == self.n_features, f'Expected {self.n_features} features, got {label_dim}'

        torch_dtype = model_with_gradiend.gradiend.torch_dtype

        device = model_with_gradiend.gradiend.device_encoder
        encoded = []

        if eval_batch_size > 1:
            for i in range(0, len(grads), eval_batch_size):
                batch = grads[i:min(i + eval_batch_size, len(grads))]
                batch_on_device = [g.to(device, dtype=torch_dtype) for g in batch]
                encoded_values = model_with_gradiend.gradiend.encoder(torch.stack(batch_on_device))
                encoded.extend(encoded_values.tolist())
                # free memory on device
                del batch_on_device
                torch.cuda.empty_cache()  # if using GPU, it helps clear memory
        else:
            for grads in grads:
                encoded_value = model_with_gradiend.gradiend.encoder(grads.to(device, dtype=torch_dtype))
                encoded.append(encoded_value.item())

        if hasattr(labels[0], '__getitem__'):
            # if labels are tuples or lists, we need to extract the first element
            scores = [pearsonr([l[i] for l in labels], [e[i] for e in encoded]).correlation for i in range(self.n_features)]
            score = np.mean(scores)
        else:
            score = pearsonr(labels, encoded).correlation

        if np.isnan(score):
            score = 0.0

        # split the encoded values by label value
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()

        if isinstance(labels[0], list):
            labels = [tuple(label) for label in labels]

        classes = set(labels)

        encoded_values_by_class = {label: [e for e, l in zip(encoded, labels) if l == label] for label in classes}
        mean_encoded_values = {label: np.mean(values) for label, values in encoded_values_by_class.items()}

        encoded = [(e, (label,)) for e, label in zip(encoded, labels)]

        end = time.time()

        print(f'Evaluated in {(end - start):.2f}s, {mean_encoded_values}')

        for label, encoded_values in encoded_values_by_class.items():
            print(f'Class {label} mean encoded value: {encoded_values[:10]}...')

        # todo check if labels have known names in GRADIEND (like 0='F', 1='M', etc.), then also add these names as keys in the result dict
        return {
            'score': score,
            'encoded': encoded,
            'encoded_by_class': encoded_values_by_class,
            'mean_by_class': mean_encoded_values,

        }

    def analyze_decoder(self, path_or_model, part='decoder', top_k_part='decoder', top_k=None, plot=False, larger_is_better=True, **kwargs):
        if isinstance(path_or_model, str):
            model_with_gradiend = ModelWithGradiend.from_pretrained(path_or_model)
        else:
            model_with_gradiend = path_or_model


        data = self.evaluate_gradiend(model_with_gradiend, top_k=top_k, part=part, top_k_part=top_k_part)

        #if plot:
        #    model = os.path.basename(model.removesuffix('model_with_gradiend'))
        #    plot_bert_with_ae_results(data, model, feature_factors=feature_factors, lrs=lrs, thorough=large, **kwargs)

        aggregation = max if larger_is_better else min
        raw_results = data.copy()
        relevant_results = data.copy()
        for key in self.metric_keys:
            arg_max = aggregation(raw_results, key=lambda x: raw_results[x][key])
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

    def evaluate_gradiend(self,
                          model,
                          feature_factors=(-1.0, 0.0, 1.0),
                          lrs=(1e-5, 1e-4, 1e-3),
                          top_k=None,
                          part='decoder',
                          top_k_part='decoder',
                          **kwargs,
                          ):
        raise NotImplementedError("Subclasses should implement this method to evaluate the model with GRADIEND.")

    def evaluate(self, model_with_gradiend, eval_data, eval_batch_size=32, max_subset_size=5, verbose=True, **kwargs):
        start = time.time()
        single_eval = False

        if isinstance(eval_data, ModelFeatureTrainingDataset):
            eval_data = {'main': eval_data}
            single_eval = True
        else:
            if not isinstance(eval_data, dict):
                raise TypeError("eval_data must be a dictionary.")

            if 'gradients' in eval_data and 'labels' in eval_data:
                eval_data = {'main': eval_data}
                single_eval = True
            else:
                # sanitize eval_data
                for k, v in eval_data.items():
                    if 'gradients' not in v or 'labels' not in v:
                        raise ValueError(f"eval_data must contain 'gradients' and 'labels' keys, but did not found these for {k}.")

        # sort eval_data s.t. 'train' gets evaluated before 'main'
        eval_data = dict(sorted(eval_data.items(), key=lambda x: x[0] != 'train'))

        torch_dtype = model_with_gradiend.gradiend.torch_dtype
        device = model_with_gradiend.gradiend.device_encoder
        source = kwargs['config']['source']
        target_key = f'{source}_target'
        total_result = {}
        other_interesting_keys = ['texts', 'factual_target', 'counterfactual_target']
        targets2indices = defaultdict(list)
        other_keys = defaultdict(list)
        for eval_key, eval_sub_data in eval_data.items():

            encoded = []
            binary_labels = []
            if isinstance(eval_sub_data, ModelFeatureTrainingDataset):
                labels = []
                for i, entry in enumerate(tqdm(eval_sub_data, desc=f'Encoding eval data for {eval_key}', leave=False)):
                    grad = entry['source']
                    label = entry['label']
                    encoded_value = model_with_gradiend.encode(grad.to(device, dtype=torch_dtype))
                    encoded.append(encoded_value.tolist())
                    labels.append(label)
                    targets2indices[target_key].append(i)
                    binary_labels.append(entry['binary_label'])

                    for key in other_interesting_keys:
                        if key in entry:
                            other_keys[key].append(entry[key])
            else:
                grads = eval_sub_data['gradients']
                labels = eval_sub_data['labels']

                # Encode gradients
                if eval_batch_size > 1:
                    for i in range(0, len(grads), eval_batch_size):
                        batch = grads[i:min(i + eval_batch_size, len(grads))]
                        batch_on_device = [g.to(device, dtype=torch_dtype) for g in batch]
                        encoded_batch = model_with_gradiend.encode(torch.stack(batch_on_device))
                        encoded.extend(encoded_batch.tolist())
                        del batch_on_device
                        del encoded_batch
                        torch.cuda.empty_cache()
                else:
                    for g in grads:
                        encoded_value = model_with_gradiend.encode(g.to(device, dtype=torch_dtype))
                        encoded.append(encoded_value.tolist())
                        del encoded_value

                for i, t in enumerate(eval_sub_data[target_key]):
                    targets2indices[t].append(i)

                other_keys = {}
                for key in other_interesting_keys:
                    if key in eval_sub_data:
                        other_keys[key] = eval_sub_data[key]



                binary_labels = eval_sub_data['binary_labels']

            encoded = np.array(encoded).tolist()  # ensure JSON-serializable format
            labels = np.array(labels).tolist()
            encoded_dim = len(encoded[0]) if isinstance(encoded[0], list) else 1
            best_dims = None  # to store best performing subset
            score = 0.0
            subset_results = None
            multi_output = None

            # Try direct dimensional match
            labels_np = np.array(labels)
            encoded_np = np.array(encoded)


            if len(labels_np.shape) > 1 and labels_np.shape[1] > 1 and encoded_np.shape[1] == 1:
                # labels are categorical and encoded is scalar
                def correlation_ratio(categories, values):
                    categories = np.array(categories)
                    values = np.array(values)
                    class_means = [values[categories == c].mean() for c in np.unique(categories)]
                    n = [np.sum(categories == c) for c in np.unique(categories)]
                    grand_mean = values.mean()
                    ss_between = np.sum(
                        [n[i] * (class_means[i] - grand_mean) ** 2 for i in range(len(class_means))])
                    ss_total = np.sum((values - grand_mean) ** 2)
                    return np.sqrt(ss_between / ss_total)

                # todo make this generic; this is just a simple fix
                category_mapper = {
                    (-1, 0, 0): 'White',
                    (0, -1, 0): 'White',
                    (1, 0, 0): 'Black',
                    (0, 0, -1): 'Black',
                    (0, 1, 0): 'Asian',
                    (0, 0, 1): 'Asian',
                }
                categories = [category_mapper.get(tuple(row), 'Other') for row in labels_np]

                final_score = correlation_ratio(categories, encoded_np[:, 0])
            else:
                if len(labels_np.shape) == 2:
                    n_samples, label_dim = labels_np.shape
                elif len(labels_np.shape) == 1:
                    n_samples = len(labels_np)
                    label_dim = 1
                    labels_np = labels_np.reshape(-1, 1)
                else:
                    raise ValueError("Labels must be 1D or 2D array-like structure.")
                _, encoding_dim = encoded_np.shape

                if n_samples != encoded_np.shape[0]:
                    raise ValueError("Number of samples does not match between labels and encodings")

                # Initialize correlation matrix: label_dim × encoding_dim
                correlation_matrix = np.full((label_dim, encoding_dim), np.nan)

                # Compute correlations
                for i in range(label_dim):
                    for j in range(encoding_dim):
                        corr = pearsonr(labels_np[:, i], encoded_np[:, j]).correlation
                        if not np.isnan(corr):
                            correlation_matrix[i, j] = corr

                try:
                    # Find best match for each label dimension
                    best_encoding_dims = np.nanargmax(np.abs(correlation_matrix), axis=1).tolist()
                    best_scores = correlation_matrix[np.arange(label_dim), best_encoding_dims].tolist()

                    # Optional: include signs, abs scores, or full matches
                    result_per_label_dim = [
                        {
                            "label_dim": int(i),
                            "best_encoding_dim": int(j),
                            "correlation": float(correlation_matrix[i, j]),
                            "abs_correlation": float(abs(correlation_matrix[i, j]))
                        }
                        for i, j in enumerate(best_encoding_dims)
                    ]
                except Exception as e:
                    print(f"Error finding best encoding dimensions: {e}")
                    result_per_label_dim = []
                    best_encoding_dims = []
                    best_scores = []

                # Final summary score
                final_score = float(np.nanmean(np.abs(best_scores)))

                # Output: for programmatic and human consumption
                multi_output = {
                    "correlation_matrix": correlation_matrix.tolist(),
                    "best_encoding_dims": best_encoding_dims,
                    "best_scores": best_scores,
                    "result_per_label_dim": result_per_label_dim,
                    "final_score": final_score
                }
            score = final_score

            if eval_key == 'main':

                if 'train' in eval_data:
                    encoded_train_np = np.array(total_result['train']['encoded'])
                    labels_train_np = np.array(eval_data['train']['labels'])
                    encoded_test_np = encoded_np
                    labels_test_np = labels_np
                else:
                    # apply deterministic train test split to data
                    encodings_train_np, encodings_test_np, labels_train_np, labels_test_np = train_test_split(
                        encoded_np, labels_np, test_size=0.2, random_state=42
                    )


                def is_binary_column(col):
                    unique = np.unique(col)
                    return len(unique) == 2 and set(unique) <= {0, 1}

                def analyze_label_encodings(
                        labels_train: np.ndarray,
                        labels_test: np.ndarray,
                        encoded_train: np.ndarray,
                        encoded_test: np.ndarray,
                        subset_sizes = [2]
                ):
                    n_labels = labels_train.shape[1]
                    n_enc_dims = encoded_train.shape[1]

                    results = []

                    for i in range(n_labels):
                        y_train = labels_train[:, i]
                        y_test = labels_test[:, i]
                        is_binary = is_binary_column(y_train)

                        best_score = -np.inf
                        best_subset = None
                        best_coef = None
                        best_model = None

                        for k in subset_sizes:
                            for dims in combinations(range(n_enc_dims), r=k):
                                X_train_sub = encoded_train[:, dims]
                                X_test_sub = encoded_test[:, dims]

                                try:
                                    if is_binary:
                                        model = LogisticRegression().fit(X_train_sub, y_train)
                                        y_pred = model.predict(X_test_sub)
                                        score = accuracy_score(y_test, y_pred)
                                    else:
                                        model = LinearRegression().fit(X_train_sub, y_train)
                                        y_pred = model.predict(X_test_sub)
                                        score = r2_score(y_test, y_pred)
                                except Exception as e:
                                    # Skip ill-conditioned subsets
                                    continue

                                if score > best_score:
                                    best_score = score
                                    best_subset = dims
                                    best_coef = model.coef_
                                    best_model = model

                        results.append({
                            'label_dim': i,
                            'is_binary': is_binary,
                            'best_subset': best_subset,
                            'score': best_score,
                            'coef': best_coef.tolist() if best_coef is not None else None,
                            'intercept': (best_model.intercept_.tolist() if best_model.intercept_ is not None else None) if best_model else None,
                            'subset_size': len(best_subset) if best_subset else None,
                        })
                        return results
                try:
                    subset_results = analyze_label_encodings(labels_train_np, labels_test_np, encoded_train_np, encoded_test_np)
                except Exception as e:
                    pass




            encoded_by_target = defaultdict(list)
            for target, indices in targets2indices.items():
                encoded_by_target[target].extend([encoded[i] for i in indices])
            encoded_by_target = dict(encoded_by_target)

            # Group encoded values by class

            if isinstance(binary_labels[0], list):
                binary_labels = [tuple(l) for l in binary_labels]

            # todo binary_labels should actually never be np.bool_ but it is???
            if isinstance(binary_labels[0], np.bool_):
                binary_labels = [bool(l) for l in binary_labels]

            if isinstance(binary_labels[0], bool):
                class_labels = labels
            else:
                class_labels = binary_labels

            encoded_by_class = {}
            for enc, label in zip(encoded, class_labels):
                if isinstance(label, (list, tuple)):
                    key = tuple(label)
                elif isinstance(label, (float, int, bool)):
                    key = label
                elif isinstance(label, (np.bool_, np.float32, np.int32)):
                    key = label.item()
                else:
                    raise ValueError(f"Unsupported label type: {type(label)}")

                if key not in encoded_by_class:
                    encoded_by_class[key] = []
                encoded_by_class[key].append(enc)

            # Compute means (convert to lists to be JSON safe)
            mean_by_class = {
                label: [float(v) for v in np.mean(vals, axis=0)]
                for label, vals in encoded_by_class.items()
            }

            #encoded_combined = [(e, label) for e, label in zip(encoded, labels)]

            end = time.time()
            if verbose:
                print(f"Evaluated in {(end - start):.2f}s. Score: {score:.4f}")
                if best_dims:
                    print(f"Best performing dimensions: {best_dims}")

            result = {
                'score': float(score),
                'encoded': encoded,
                'labels': labels,
                'binary_labels': binary_labels,
                'encoded_by_class': {str(k) if isinstance(k, (tuple, list)) else k: v for k, v in encoded_by_class.items()},
                'mean_by_class': {str(k) if isinstance(k, (tuple, list)) else k: v for k, v in mean_by_class.items()},
                'best_dims': best_dims,
                'encoded_by_target': encoded_by_target,
            }

            if subset_results:
                result['subset_results'] = subset_results

            if multi_output:
                result['sub_results'] = multi_output

            for key in ['texts', 'factual_target', 'counterfactual_target']:
                if key in other_keys:
                    result[key] = other_keys[key]

            total_result[eval_key] = result


        # store evaluation results
        global_step = kwargs['training_stats']['global_step']
        base_output = kwargs['config']['output']
        output_eval_results = f'{base_output}/eval_training/{global_step}.json'
        os.makedirs(os.path.dirname(output_eval_results), exist_ok=True)
        json.dump(total_result, open(output_eval_results, 'w'), indent=2)

        # release GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        if single_eval:
            # if we evaluated a single dataset, return the result directly
            return total_result['main']


        return total_result

    py_print = print

    def select(self,
               model,
               max_size=None,
               print=True,
               force=True,
               plot=True,
               output_suffix="",
               output=True
               ):
        model_base_name = os.path.basename(model)
        output_result = f'results/models/{self.id}/evaluation_{model_base_name}.json'
        model_with_gradiend = None

        if force or not os.path.isfile(output_result):
            split = 'test'
            file_format = 'csv'
            enc_output = get_file_name(model, max_size=max_size, file_format=file_format, split=split, v=3)
            if not os.path.isfile(enc_output):
                py_print(f'Analyze model {model} since file {enc_output} does not exist')
                analysis = self.analyze_models(model, max_size=max_size, split=split, force=force)
            try:
                encoder_metrics = get_model_metrics(enc_output)
            except FileNotFoundError as e:
                py_print(f'Error loading encoder metrics from {enc_output}: {e}')
                encoder_metrics = {}

            decoder_metrics = self.analyze_decoder(model, large=True, plot=plot, larger_is_better=self.larger_is_better)

            model_with_gradiend = ModelWithGradiend.from_pretrained(model)

            result = {
                'encoder': encoder_metrics,
                'decoder': decoder_metrics,
                'training': model_with_gradiend.gradiend.kwargs['training']
            }

            # create biased models
            base_model_output = f'results/changed_models/{model_base_name}'
            if force or not os.path.isdir(base_model_output):
                model_with_gradiend.base_model.save_pretrained(base_model_output)
                model_with_gradiend.tokenizer.save_pretrained(base_model_output)

                for key in self.metric_keys:
                    key_metrics = decoder_metrics[key]
                    lr = key_metrics['lr']
                    feature_factor = key_metrics['feature_factor']

                    changed_model = model_with_gradiend.modify_model(lr=lr, feature_factor=feature_factor)
                    version = self.version_map.get(key, key)
                    key_output = f'{base_model_output}-{version}{output_suffix}'
                    changed_model.save_pretrained(key_output)
                    model_with_gradiend.tokenizer.save_pretrained(key_output)
                    py_print(
                        f'Saved {key} model to {key_output} with gender factor {feature_factor} and learning rate {lr}')

                    change_stats = {
                        'feature_factor': feature_factor,
                        'lr': lr,
                        'version': version,
                        'model': key_output,
                        'base_model': base_model_output,
                        'encoder_metrics': encoder_metrics,
                        'decoder_metrics': key_metrics,
                        'training': model_with_gradiend.gradiend.kwargs['training']

                    }
                    change_stats_file = f'{key_output}/stats.json'
                    with open(change_stats_file, 'w') as f:
                        json.dump(change_stats, f, indent=2)

                    del changed_model
                    # release memory
                    torch.cuda.empty_cache()

            json_compatible_result = convert_tuple_keys_to_strings(result)
            with open(output_result, 'w') as f:
                json.dump(json_compatible_result, f, indent=2)
        else:
            with open(output_result, 'r') as f:
                result = json.load(f)
                encoder_metrics = result['encoder']
                decoder_metrics = result['decoder']

        if output and not force:
            if model_with_gradiend is None:
                model_with_gradiend = ModelWithGradiend.from_pretrained(model)

            # save the best models to output
            for key in self.metric_keys:
                key_metrics = decoder_metrics[key]
                lr = key_metrics['lr']
                feature_factor = key_metrics['feature_factor']
                changed_model = model_with_gradiend.modify_model(lr=lr, feature_factor=feature_factor)

                output_path = f'results/changed_models/{model_base_name}-{self.version_map.get(key, key)}{output_suffix}'
                if not os.path.isdir(output_path):
                    changed_model.save_pretrained(output_path)
                    model_with_gradiend.tokenizer.save_pretrained(output_path)
                    py_print(f'Saved {key} model to {output_path}')

                    change_stats = {
                        'feature_factor': feature_factor,
                        'lr': lr,
                        'version': key,
                    }
                    change_stats_file = f'{key_output}/stats.json'
                    with open(change_stats_file, 'w') as f:
                        json.dump(change_stats, f, indent=2)
                else:
                    py_print(f"Model {output_path} already exists")
                    # check if saved model is the same
                    saved_model = AutoModelForLM.from_pretrained(output_path)
                    # todo adjust for generative models!
                    try:
                        if not np.allclose(saved_model.base_model.embeddings.position_embeddings.weight.cpu().detach(),
                                           changed_model.base_model.embeddings.position_embeddings.weight.cpu().detach()):
                            py_print(f'Error: Existing Model {output_path} was not the same as the current model')
                            changed_model.save_pretrained(output_path)
                    except AttributeError:
                        py_print(f'WARNING: Model {output_path} does not have position embeddings, skipping check')

        if print:
            py_print(f'Evaluation for model {model}')
            py_print('Encoder:')
            if 'acc_total' in encoder_metrics:
                py_print('\tAccuracy Total:', encoder_metrics['acc_total'])
                py_print('\tCorrelation:', encoder_metrics['pearson_total'])
                py_print('\tAccuracy:', encoder_metrics['acc'])
            if 'pearson' in encoder_metrics:
                py_print('\tCorrelation MF:', encoder_metrics['pearson'])
            elif 'pearson_MF' in encoder_metrics:
                py_print('\tCorrelation MF:', encoder_metrics['pearson'])

#            py_print('\tMA', encoder_metrics['encoded_abs_means'])
            py_print('Decoder:')
            for key in self.metric_keys:
                py_print(f'\t{key}:', decoder_metrics[key])
                py_print(f'\tBase model {key}', decoder_metrics['base'][key])

        return result

    def post_training(self, model_with_gradiend, **kwargs):
        """
        Post-training evaluation method to be implemented by subclasses.
        This method can be used to perform additional evaluations after training.
        """
        global_step = model_with_gradiend.gradiend.kwargs['training']['global_step']
        base_output = model_with_gradiend.gradiend.kwargs['training']['config']['output']
        return self._post_training(base_output, step=global_step)



import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class BatchedTrainingDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 batch_size,
                 batch_criterion,
                 max_size=None,
                 seed=42,
                 shuffle_batches=None,
                 max_length=256,
                 balance_column=None,
                 shuffle_within=None,
                 ):
        assert batch_size > 0, "batch_size must be set and > 0"
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id
        self.batch_size = batch_size
        self.batch_criterion = batch_criterion
        self.seed = seed
        self.shuffle_batches = shuffle_batches or self.batch_size > 1
        self.max_length = max_length
        self.balance_column = balance_column
        self.shuffle_within = shuffle_within or self.batch_size > 1
        rng = random.Random(seed)

        if max_size is not None:
            data = data.sample(n=max_size, random_state=seed).reset_index(drop=True)
        else:
            data = data.reset_index(drop=True)
        self.data = data

        # ------------------------------------------------------
        # Step 1: split into balance groups
        # ------------------------------------------------------
        if balance_column is not None:
            grouped_by_balance = dict(tuple(data.groupby(balance_column)))
        else:
            grouped_by_balance = {"__all__": data}

        self.group_batches = {}
        total_batches = 0

        # ------------------------------------------------------
        # Step 2: inside each balance group, apply batch_criterion
        # ------------------------------------------------------
        for gname, gdata in grouped_by_balance.items():
            # determine grouping keys
            bc = batch_criterion
            if bc == "source_target":
                bc = ["source", "target"]
                group_keys = gdata[bc].apply(lambda row: tuple(row), axis=1)
            elif isinstance(bc, str):
                group_keys = gdata[bc]
            elif callable(bc):
                group_keys = gdata.apply(bc, axis=1)
            elif isinstance(bc, list):
                group_keys = gdata[bc].apply(lambda row: tuple(row), axis=1)
            else:
                raise TypeError("batch_criterion must be str, list, or callable")

            grouped = gdata.groupby(group_keys)

            batches = []
            for _, group in grouped:
                if len(group) < batch_size:
                    continue
                if self.shuffle_within:
                    group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
                n_full = len(group) // batch_size
                for i in range(n_full):
                    batches.append(group.iloc[i * batch_size:(i + 1) * batch_size])

            if self.shuffle_within:
                rng.shuffle(batches)

            self.group_batches[gname] = batches
            total_batches += len(batches)

        self.balance_keys = list(self.group_batches.keys())
        self.total_batches = total_batches
        if self.balance_column and self.batch_size > 1:
            self.total_samples = 100 * self.total_batches * batch_size
            # todo think of something smarter here to ensure enough samples per group
            # for now, this hack works since we stop after some number of samples anyway rather than training for epochs
        else:
            self.total_samples = total_batches * batch_size

        print(f"BalancedBatchedTrainingDataset created with {self.total_batches} batches "
              f"({self.total_samples} samples) across {len(self.balance_keys)} balance groups.")



    def reshuffle(self):
        """Reshuffle batches inside each balance group (call at each epoch)."""
        rng = random.Random(self.seed)
        for gname, batches in self.group_batches.items():
            rng.shuffle(batches)

    def __len__(self):
        return self.total_samples


    def __getitem__(self, idx):
        # 1. Global batch index and offset inside the batch
        batch_idx = idx // self.batch_size
        in_batch_offset = idx % self.batch_size

        # 2. Choose which balance group to use (round-robin)
        if self.balance_column is None:
            gname = self.balance_keys[0]
        else:
            gname = self.balance_keys[batch_idx % len(self.balance_keys)]

        # 3. Determine which batch inside this group to use
        group_batches = self.group_batches[gname]
        local_batch_idx = batch_idx // len(self.balance_keys)

        # If the group has fewer batches, wrap around instead of crashing
        if local_batch_idx >= len(group_batches):
            local_batch_idx = local_batch_idx % len(group_batches)

        batch = group_batches[local_batch_idx]

        # 4. Pick the row inside the batch
        row = batch.iloc[in_batch_offset]
        return row

# todo soften the requirements s.t. also a few texts without [MASK] tokens can be used during training to reduce the token length and computational time overall
    def _create_item(self, text: str, target: str):
        # Replace a single [MASK] with correct number of [MASK] tokens
        if not self.is_generative and self.mask_token not in text:
            raise ValueError("Input text must contain at least one [MASK] token placeholder.")

        mask_count = 0 if self.is_generative else text.count(self.mask_token)
        if mask_count > 1:
            raise ValueError("Input text must contain exactly one [MASK] token placeholder.")

        target_tokens = self.tokenizer(target, add_special_tokens=False)['input_ids']
        num_target_tokens = len(target_tokens)

        if self.is_generative:
            expanded_text = text
        else:
            # Replace the single [MASK] with the correct number of [MASK] tokens
            mask_tokens_str = ' '.join([self.mask_token] * num_target_tokens)
            expanded_text = text.replace(self.mask_token, mask_tokens_str)

        # Tokenize the expanded input
        encoded = self.tokenizer(expanded_text,
                                 return_tensors='pt',
                                 add_special_tokens=True,
                                 truncation=True,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 )
        input_ids = encoded['input_ids']
        #input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask']
        #attention_mask = encoded['attention_mask'][0]

        # Identify [MASK] token positions
        labels = torch.full_like(input_ids, -100)
        if self.is_generative:
            #mask_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            last_idxs = attention_mask.sum(dim=1)
            if target in self.tokenizer.vocab:
                target_idx = self.tokenizer.vocab[target]
            else:
                target_encoded = self.tokenizer(target, add_special_tokens=False)
                #if len(target_encoded['input_ids']) != 1:
                #    raise ValueError(f"Target '{target}' is not a single token in the tokenizer vocabulary.")

                target_idx = target_encoded['input_ids'][0]
            #last_idx = (input_ids != self.tokenizer.pad_token_id).nonzero()
            #if len(last_idx) > 0:
            #    mask_token_mask[tuple(last_idx[-1].tolist())] = True
            #    labels[mask_token_mask] = input_ids[mask_token_mask]
            for i, idx in enumerate(last_idxs):
                if idx < input_ids.size(1):
                    labels[i, idx] = target_idx
        else:
            mask_token_id = self.tokenizer.convert_tokens_to_ids(self.mask_token)
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

            # Create labels: all -100 except at mask positions
            for pos, tok in zip(mask_positions, target_tokens):
                labels[pos] = tok

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

plural_keys = {
    'gradient': 'gradients',
    'source': 'gradients',
    'text': 'texts',
    'label': 'labels',
    'binary_label': 'binary_labels',
    'metadata': 'metadata',
}

class JITGradientLoader:
    def __init__(self, ds):
        self.ds = ds

    def __iter__(self):
        for i in range(len(self.ds)):
            d = self.ds[i]
            del d['target']


            # convert in dict of lists
            results = {}
            for key in d.keys():
                key_plural = plural_keys[key] if key in plural_keys else key
                results[key_plural] = [entry[key] for entry in [d]]

            yield results

def create_eval_dataset(setup, model_with_gradiend, split='val', source='factual', dataset_filter=None, use_caching=False, pre_load_gradients=True, **kwargs):
    if use_caching is None and hasattr(model_with_gradiend, 'feature_creator_id'):
        use_caching = model_with_gradiend.feature_creator_id == 'grad'

    start = time.time()
    dataset = setup.create_training_data(model_with_gradiend.tokenizer, split=split, batch_size=1, **kwargs)

    if dataset_filter is not None:
        dataset = dataset_filter(dataset)

    base_model = model_with_gradiend.base_model.name_or_path
    base_model = os.path.basename(base_model)
    layers_hash = model_with_gradiend.layers_hash
    if use_caching:
        cache_dir = f'results/cache/gradients/{setup.id}/{base_model}/{layers_hash}'
    else:
        cache_dir = None

    gradient_dataset = ModelFeatureTrainingDataset(
        training_data=dataset,
        tokenizer=model_with_gradiend.tokenizer,
        feature_creator=model_with_gradiend.feature_creator,
        feature_creator_id=model_with_gradiend.feature_creator_id,
        source=source,
        target=None,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device=torch.device('cpu') if pre_load_gradients else model_with_gradiend.gradiend.device_encoder,
        return_metadata=True,
    )

    if not pre_load_gradients:
        return gradient_dataset


    result_dicts = []

    for entry in tqdm(gradient_dataset, desc=f'Loading cached evaluation data', leave=False):
        del entry['target']
        result_dicts.append(entry)

    # convert in dict of lists
    results = {}
    for key in result_dicts[0].keys():
        key_plural = plural_keys[key] if key in plural_keys else key
        results[key_plural] = [entry[key] for entry in result_dicts]

    print(f'Loaded the evaluation data with {len(result_dicts)} entries in {time.time() - start:.2f}s')
    return results
