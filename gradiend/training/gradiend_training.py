import json
import os
import shutil
import time

import numpy as np
import torch

from gradiend.evaluation.analyze_encoder import get_model_metrics
from gradiend.setups.gender.en import GenderEnSetup
from gradiend.training import train_all_layers_gradiend, train_multiple_layers_gradiend, PolarFeatureLoss


def train(setup, base_model, model_config=None, n=3, metric='pearson', force=False, version=None, clear_cache=False, only_return_output=False):
    model_config = model_config or {}

    if version is None or version == '':
        version = ''
    else:
        version = f'/{version}'


    base_model_id = (base_model if isinstance(base_model, str) else base_model.name_or_path).removesuffix('/best').split('/')[-1]
    main_output = f'results/models/{setup.id}/{base_model_id}{version.replace("/", "-")}'
    if only_return_output:
        return main_output

    metrics = []
    total_start = time.time()
    times = []
    init_model_config = model_config.copy()

    for i in range(n):
        start = time.time()
        output = f'results/experiments/gradiend/{setup.id}/{base_model_id}{version}/{i}'
        metrics_file = f'{output}/metrics.json'
        if not force and os.path.exists(metrics_file):
            metrics.append(json.load(open(metrics_file)))
            print(f'Skipping training of {output} as it already exists')

            continue

        if not os.path.isfile(f'{output}/pytorch_model.bin') or force:
            print('Training', output)
            model_config['seed'] = i + init_model_config.get('seed', 0)
            if 'layers' in model_config:
                train_multiple_layers_gradiend(setup, model=base_model, output=output, **model_config)
            else:
                train_all_layers_gradiend(setup, model=base_model, output=output, **model_config)
        else:
            print('Model', output, 'already exists, skipping training, but evaluate')

        try:
            setup.analyze_models(output, split='val', force=force)
            model_metrics = setup.get_model_metrics(output, split='val')
            metric_value = model_metrics[metric]
            json.dump(metric_value, open(metrics_file, 'w'))
            metrics.append(metric_value)
        except Exception as e:
            print(f'Error analyzing model {output}, {e}, skipping...')

        times.append(time.time() - start)

        if clear_cache:
            cache_folder = f'results/cache/gradients/{setup.id}/{base_model_id}'
            if os.path.exists(cache_folder):
                shutil.rmtree(cache_folder)

    if metrics:
        print(f'Metrics for model {base_model}: {metrics}')
        best_index = np.argmax(metrics)
        print('Best metric at index', best_index, 'with value', metrics[best_index])

        # copy the best model to output
        shutil.copytree(f'results/experiments/gradiend/{setup.id}/{base_model_id}{version}/{best_index}', main_output, dirs_exist_ok=True)
        print('Copied best model to', main_output)

        total_time = time.time() - total_start
        if times:
            print(f'Trained {len(times)} models in {total_time}s')
            print(f'Average time per model: {np.mean(times)}')
        else:
            print('All models were already trained before!')
    elif n == 1:
        print(f'No metrics found for model {base_model}, but trained once')
        # check if copying is necessary
        if not os.path.exists(main_output) or not os.path.isfile(f'{main_output}/pytorch_model.bin'):
            print('Copying trained model to output')
            shutil.copytree(f'results/experiments/gradiend/{setup.id}/{base_model_id}{version}/0', main_output, dirs_exist_ok=True)
    else:
        print(f'No metrics found for model {base_model_id}, skipping saving output')
        main_output = output


    return main_output

def train_for_configs(setup, model_configs, n=3, metric='pearson', force=False, version=None, clear_cache=False, selecting_after_total_training=False):
    models = []
    for base_model, model_config in model_configs.items():
        model = train(setup, base_model, model_config, only_return_output=False, n=n, metric=metric, force=force, version=version, clear_cache=clear_cache)
        models.append(model)
        if not selecting_after_total_training:
            try:
                setup.select(model)
                pass
            except NotImplementedError as e:
                print(f'Error selecting model: {e}')


    if selecting_after_total_training:
        try:
            for model in models:
                setup.select(model)
        except Exception as e:
            print(f'Error selecting models: {e}')

    return models


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
        model = train(setup, base_model, model_config, n=3, version='test2', clear_cache=False, force=False)
        models.append(model)

    for model in models:
        setup.select(model)
