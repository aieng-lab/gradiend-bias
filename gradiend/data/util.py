import json
import os

import inflect
import pandas as pd


def get_file_name(base_file_name, file_format=None, **kwargs):
    base_name = os.path.basename(base_file_name)
    output = str(base_file_name)
    if '.' in base_name[-5:]:
        current_file_format = base_name.split('.')[-1]
        if current_file_format in {'csv', 'json', 'txt', 'tsv'}:
            if current_file_format != file_format:
                raise ValueError(f'Provided format of file {current_file_format} does not match key word argument {file_format}: {base_file_name}')
            output = base_file_name[:-len(file_format) - 1]

    first_param = True
    for key, value in sorted(kwargs.items()):
        if key in {'top_k', 'top_k_part'} and value is None:
            continue

        if value is not None:
            if first_param:
                output += '_params'
                first_param = False

            output += f'_{key[:3]}_{value}'

    if file_format and not output.endswith(file_format):
        output += '.' + file_format

    return output

def sanitize_split(split):
    if split == 'val':
        return 'validation'
    return split

def json_loads(x):
    if isinstance(x, (float, int)):
        return x

    try:
        return json.loads(x)
    except Exception:
        return [xx.removeprefix("'").removesuffix("'") for xx in x.removeprefix('[').removesuffix(']').split(',')]

def json_dumps(x):
    if isinstance(x, (float, int, str)):
        return x

    if isinstance(x, list) and len(x) == 1:
        return x[0]

    return json.dumps(x)

def enrich_with_plurals(input_dict):
    # Create an inflect engine
    p = inflect.engine()

    # Iterate over the dictionary values
    for key, value in input_dict.items():
        # Enrich the list with plural forms
        plural_values = [p.plural(word) for word in value]
        # Update the dictionary with the enriched list
        input_dict[key] = list(set(plural_values + input_dict[key]))

    return input_dict
