import itertools

import numpy as np
import pandas as pd
from datasets import Dataset

from gradiend.setups.util.multiclass import MultiClassSetup, Bias3DCombinedSetup, \
    TrainingDataset
from gradiend.training.gradiend_training import train_for_configs




def create_training_dataset(tokenizer,
                            max_size=None,
                            batch_size=1,
                            neutral_data=False,
                            split='train',
                            neutral_data_prop=0.5,
                            classes=None,
                            bias_type='race',
                            single_word_texts=False,
                            is_generative=None,
                            return_training_dataset=True,
                            ):
    if classes is None or len(classes) < 2:
        raise ValueError("Please provide at least two classes in the 'classes' list.")

    all_dfs = []
    label_counter = 0
    is_generative = is_generative or tokenizer.mask_token_id is None

    vocab = set(tokenizer.get_vocab().keys())

    pair_to_index = {
        frozenset((i, j)): idx
        for idx, (i, j) in enumerate(itertools.combinations(range(len(classes)), 2))
    }
    num_pairs = len(pair_to_index)
    df_ctr = 0

    # Load all pairwise datasets
    for i, class_from in enumerate(classes):
        for j, class_to in enumerate(classes):
            if i == j:
                continue  # skip same-class
            ds_path = f"data/{bias_type}/{class_from}_to_{class_to}"
            try:
                df = Dataset.load_from_disk(ds_path).to_pandas()
            except FileNotFoundError:
                continue  # skip if dataset doesn't exist

            pair_idx = pair_to_index[frozenset((i, j))]

            # add index label
            if num_pairs == 1:
                df["label_idx"] = label_counter
                label_counter += 1
            else:
                df["label_idx"] = pair_idx

            # add one-hot columns
            one_hot = np.zeros(num_pairs, dtype=int)
            one_hot[pair_idx] = 1 if i > j else -1
            for k in range(num_pairs):
                df[f"label_{k}"] = one_hot[k]


            # Deterministic split
            if split:
                # todo save the splits persistently
                train, val, test = 0.7, 0.2, 0.1
                df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                n_train = int(len(df) * train)
                n_val = int(len(df) * val)
                if split == "train":
                    df = df[:n_train]
                elif split == "val":
                    df = df[n_train:n_train + n_val]
                elif split == "test":
                    df = df[n_train + n_val:]
                else:
                    raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")


            if is_generative:
                # drop entries where [MASK] appears within the first 10 words to ensure enough context
                first_10_tokens = df['masked'].str.split().str[:10].str.join(' ')
                mask_in_first_10 = first_10_tokens.str.contains('[MASK]')
                df = df[~mask_in_first_10].reset_index(drop=True)


            is_llama = 'llama' in tokenizer.name_or_path.lower()
            if not is_llama:
                df = df[df["source"].isin(vocab) & df["target"].isin(vocab)]

            if df.empty:
                raise ValueError('No source and target words in vocab!')

            df['df_ctr'] = df_ctr
            df_ctr += 1
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No datasets found for the given classes.")

    # ensure all datasets have the same number of examples
    min_size = min(len(df) for df in all_dfs)
    all_dfs = [df.sample(min_size, random_state=42).reset_index(drop=True) for df in all_dfs]

    # Merge datasets
    templates = pd.concat(all_dfs, ignore_index=True).sample(frac=1.0).reset_index(drop=True)

    if max_size:
        # Stratified sample across all labels
        templates = templates.groupby('label_idx').apply(
            lambda x: x.sample(min(len(x), max_size // templates['label_idx'].nunique()))
        ).reset_index(drop=True)

    if num_pairs == 1:
        target_keys = 'label_idx'
    else:
        target_keys = [f'label_{i}' for i in range(num_pairs)]
# todo check that easy way to get calass is available, e.g., source_id -> categorical classes
    if return_training_dataset:
        return TrainingDataset(
            templates,
            tokenizer,
            batch_size=batch_size,
            is_generative=is_generative,
            target_key=target_keys,
        )
    return templates


class Bias1DSetup(MultiClassSetup):
    def __init__(self, bias_type, class1, class2, pretty_id=None):
        super().__init__(bias_type, class1, class2, pretty_id=pretty_id)
        self.n_features = 1

    @property
    def class1(self):
        return self.classes[0]

    @property
    def class2(self):
        return self.classes[1]

    def create_training_data(self, *args, **kwargs):
        return create_training_dataset(*args, classes=self.classes, bias_type=self.bias_type, **kwargs)

class Race1DSetup(Bias1DSetup):
    def __init__(self, class1, class2, pretty_id=None):
        super().__init__('race', class1, class2, pretty_id=pretty_id)

class WhiteBlackSetup(Race1DSetup):
    def __init__(self):
        super().__init__('white', 'black', pretty_id='Black/White')


class WhiteAsianSetup(Race1DSetup):
    def __init__(self):
        super().__init__('white', 'asian', pretty_id='Asian/White')

class BlackAsianSetup(Race1DSetup):
    def __init__(self):
        super().__init__('black', 'asian', pretty_id='Asian/Black')

class Religion1DSetup(Bias1DSetup):
    def __init__(self, religion1, religion2, pretty_id=None):
        super().__init__('religion', religion1, religion2, pretty_id=pretty_id)

class MuslimJewishSetup(Religion1DSetup):
    def __init__(self):
        super().__init__('muslim', 'jewish', pretty_id='Jewish/Muslim')

class ChristianJewishSetup(Religion1DSetup):
    def __init__(self):
        super().__init__('christian', 'jewish', pretty_id='Christian/Jewish')

class ChristianMuslimSetup(Religion1DSetup):
    def __init__(self):
        super().__init__('christian', 'muslim', pretty_id='Christian/Muslim')



class Race3DCombinedSetup(Bias3DCombinedSetup):
    def __init__(self):
        super().__init__('race', 'white', 'black', 'asian')

        self.init_gradiends = [
            f'results/models/{self.bias_type}_white_black',
            f'results/models/{self.bias_type}_white_asian',
            f'results/models/{self.bias_type}_black_asian',
        ]

class Religion3DCombinedSetup(Bias3DCombinedSetup):
    def __init__(self):
        super().__init__('religion', 'christian', 'muslim', 'jewish')

        self.init_gradiends = [
            f'results/models/{self.bias_type}_christian_jewish',
            f'results/models/{self.bias_type}_christian_muslim',
            f'results/models/{self.bias_type}_muslim_jewish',
        ]

class Race3DSetup(MultiClassSetup):
    def __init__(self):
        super().__init__('race', 'white', 'black', 'asian')
        self.n_features = 1


    def create_training_data(self, *args, **kwargs):
        return create_training_dataset(classes=self.classes, *args, **kwargs)

    def evaluate(self, model_with_gradiend, eval_data, eval_batch_size=32, max_subset_size=5, verbose=True, **kwargs):
        result = super().evaluate(model_with_gradiend, eval_data, eval_batch_size, max_subset_size, verbose, **kwargs)
#        print(result)

        encoded_by_class = result['encoded_by_class']

        # plot the encoded values by class with different colors
        import seaborn as sns
        x = []
        y = []
        for class_name, values in encoded_by_class.items():
            x.extend([class_name] * len(values))
            y.extend([v[0] for v in values])

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=x, y=y)
        plt.title(f'Encoded Values by Class for {self.pretty_id}')
        plt.xlabel('Class')
        plt.ylabel('Encoded Value')
        plt.grid(True)
        plt.show()

        return result




class Race2DSetup(MultiClassSetup):
    def __init__(self):
        super().__init__('race', 'white', 'black', 'asian')
        self.n_features = 2


    def create_training_data(self, *args, **kwargs):
        return create_training_dataset(classes=self.classes, bias_type=self.bias_type, *args, **kwargs)

    def evaluate(self, model_with_gradiend, eval_data, eval_batch_size=32, max_subset_size=5, verbose=True, **kwargs):
        result = super().evaluate(model_with_gradiend, eval_data, eval_batch_size, max_subset_size, verbose, **kwargs)
#        print(result)

        encoded_by_class = result['encoded_by_class']

        # plot the encoded values by class with different colors
        import seaborn as sns
        x = []
        y = []
        for class_name, values in encoded_by_class.items():
            x.extend([class_name] * len(values))
            y.extend([v[0] for v in values])

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=x, y=y)
        plt.title(f'Encoded Values by Class for {self.pretty_id}')
        plt.xlabel('Class')
        plt.ylabel('Encoded Value')
        plt.grid(True)
        plt.show()

        return result




class Religion3DSetup(MultiClassSetup):
    def __init__(self):
        super().__init__('religion', 'christian', 'jewish', 'muslim')
        self.n_features = 1


    def create_training_data(self, *args, **kwargs):
        return create_training_dataset(classes=self.classes, bias_type=self.bias_type, *args, **kwargs)


    def evaluate(self, model_with_gradiend, eval_data, eval_batch_size=32, max_subset_size=5, verbose=True, **kwargs):
        result = super().evaluate(model_with_gradiend, eval_data, eval_batch_size, max_subset_size, verbose, **kwargs)
#        print(result)

        encoded_by_class = result['encoded_by_class']

        # plot the encoded values by class with different colors
        import seaborn as sns
        x = []
        y = []
        for class_name, values in encoded_by_class.items():
            x.extend([class_name] * len(values))
            y.extend([v[0] for v in values])

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=x, y=y)
        plt.title(f'Encoded Values by Class for {self.pretty_id}')
        plt.xlabel('Class')
        plt.ylabel('Encoded Value')
        plt.grid(True)
        plt.show()

        return result


def train_multi_dim_gradiends(configs, version=None, activation='tanh', setups=None):
    for id, config in configs.items():
        config['activation'] = activation
        config['delete_models'] = False

    setups = setups or [WhiteBlackSetup(), WhiteAsianSetup(), BlackAsianSetup()]

    for setup in setups:
        print(f"Training setup: {setup.id}")
        #if isinstance(setup, Bias3DCombinedSetup):
        #    configs = {k: {**v, 'max_iterations': 500} for k, v in configs.items()}

        train_for_configs(setup, configs, version=version, n=1, clear_cache=True)


if __name__ == '__main__':
    configs = {
        'bert-base-cased': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=8),
        'bert-large-cased': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=4),
        'distilbert-base-cased': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff'),
        'roberta-large': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=4),

        'gpt2': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff'),
        'meta-llama/Llama-3.2-3B-Instruct': dict(eval_max_size=50, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=1, torch_dtype=torch.bfloat16, lr=1e-4),
        'meta-llama/Llama-3.2-3B': dict(eval_max_size=50, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=1, torch_dtype=torch.bfloat16, lr=1e-4),
    }

    configs_factual_source = {k: {**v, 'source': 'factual'} for k, v in configs.items()}

    all_setups = [
        ChristianJewishSetup(),
        MuslimJewishSetup(),
        WhiteAsianSetup(),
        BlackAsianSetup(),
        WhiteBlackSetup(),
        ChristianMuslimSetup(),
    ]

    try:
        train_multi_dim_gradiends(configs, version='v5', activation='tanh', setups=all_setups)
    except NotImplementedError as e:
        print(f"Error during training: {e}")
