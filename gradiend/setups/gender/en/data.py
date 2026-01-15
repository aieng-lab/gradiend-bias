from collections import defaultdict

from datasets import load_dataset as load_dataset_hf

from gradiend.setups.gender.en.names import *
from gradiend.data.util import *
from gradiend.setups.gender.en.util import gender_pronouns


def load_dataset(name, split=None, trust_remote_code=False):
    dataset = load_dataset_hf(name, split=sanitize_split(split), trust_remote_code=trust_remote_code)

    if split is None:
        return pd.concat([ds.to_pandas() for ds in dataset.values()])
    else:
        return dataset.to_pandas()

def read_namextend():
    return load_dataset('aieng-lab/namextend')

def read_namexact(split=None):
    return load_dataset('aieng-lab/namexact', split=split)

def read_genter(split=None):
    #path = 'data/genter_1.0'
    #if split:
    #    path += f'_{split}'
    #path += '.csv'
    #return pd.read_csv(path)
    return load_dataset('aieng-lab/genter', split=split, trust_remote_code=True)

def read_geneutral(max_size=None, exclude=None):
    df = load_dataset('aieng-lab/geneutral', trust_remote_code=True)

    if exclude:
        df = df[~df['text'].str.contains('|'.join(exclude), case=False, na=False)]

    if max_size:
        df = df.head(n=max_size)
    return df

def read_gentypes():
    return load_dataset('aieng-lab/gentypes')

def read_bookcorpus():
    from datasets import load_dataset
    dataset = load_dataset("bookcorpus", trust_remote_code=True)
    return dataset['train']

def get_gender_words(tokenizer=None):
    names = read_namextend()
    gender_words = [word for gender_words in read_gender_data().values() for word in gender_words]
    gender_words += names['name'].str.lower().unique().tolist()
    gender_words += gender_pronouns
    if tokenizer:
        raw_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        gender_words = list(set(token for word in gender_words for token in raw_tokenizer(word, add_special_tokens=False)['input_ids']))

    return gender_words

def read_gender_data(file_name='data/gendered_words.json', force=False, as_dict=True, include_gender_pronouns=False):

    def output(data):
        if include_gender_pronouns:
            data['M'] += ['he', 'himself', 'his', 'him']
            data['F'] += ['she', 'herself', 'her', 'hers']

        if as_dict:
            return data
        else:
            return [word for words in data.values() for word in words]

    cache_file = file_name.replace('.json', '_cache.json')
    if not force:
        try:
            with open(cache_file, 'r+') as file:
                return output(json.load(file))
        except FileNotFoundError or json.JSONDecodeError:
            pass

    with open(file_name, 'r+', encoding='utf8') as file:
        data = json.load(file)
    gender_words = defaultdict(list)
    for entry in data:
        gender = entry['gender']
        if gender in ['m', 'f']:
            word = entry['word']
            if word not in gender_pronouns and word not in gender_words[gender.upper()]:
                gender_words[gender.upper()].append(word)

                # we also consider the opposite gender word if available, e.g. 'son' -> 'daughter', because the mapping
                # in the dataset is apparently not bijective
                if 'gender_map' in entry:
                    opposite_gender = 'm' if gender == 'f' else 'f'
                    opposite_gender_word = entry['gender_map'][opposite_gender][0]['word']
                    if opposite_gender_word not in gender_words[opposite_gender.upper()]:
                        gender_words[opposite_gender.upper()].append(opposite_gender_word)

    gender_words = enrich_with_plurals(gender_words)

    with open(cache_file, 'w+') as file:
        json.dump(gender_words, file, indent=2)

    return output(gender_words)


def generate_namexact():
    return read_names_data(filter_non_unique=True,
                           minimum_count=20000,
                           gender_agreement_threshold=None,
                           filter_excluded_words=True,
                           max_entries=None,
                           subset=None)

def generate_namextend():
    return read_names_data(subset=read_namexact,  # pass unambiguous names to split wrt to the subset split!
                           filter_non_unique=False,
                           minimum_count=100,
                           gender_agreement_threshold=None,
                           filter_excluded_words=False,
                           max_entries=None)

