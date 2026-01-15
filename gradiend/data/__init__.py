import pandas as pd
from datasets import load_dataset


def read_gerneutral(max_size=None, exclude=None):
    df = load_dataset('aieng-lab/biasneutral', split='train', trust_remote_code=True).to_pandas()

    if exclude:
        df = df[~df['text'].str.contains('|'.join(exclude), case=False, na=False)]
    if max_size is not None:
        df = df.head(min(max_size, len(df)))
    return df

def read_gerneutral_local(max_size=None, exclude=None):
    df = pd.read_csv("data/gerneutral.csv")

    if exclude:
        df = df[~df['text'].str.contains('|'.join(exclude), case=False, na=False)]

    if max_size is not None:
        df = df.head(min(max_size, len(df)))
    return df