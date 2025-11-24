import gc
import random
import shutil
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

import time
import torch.nn as nn
from scipy.stats import pearsonr

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

from gradiend.training.dataset import ModelFeatureTrainingDataset

import datetime
import os

from gradiend.util import hash_it


# Create a unique directory for each run based on the current time
def get_log_dir(base_dir="logs", output=''):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_dir, output + f'_{current_time}')
    return log_dir

# Define the custom loss function
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha

    def forward(self, outputs, targets):
        mse = self.mse_loss(outputs, targets)
        l1 = self.l1_loss(outputs, targets)
        return mse + self.alpha * l1

    def __str__(self):
        return f'CombinedLoss(alpha={self.alpha})'

class PolarFeatureLoss(nn.Module):
    def __init__(self, alpha=0.001):
        super(PolarFeatureLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, output, target, encoded_value):
        mse_loss = self.mse(output, target)
        reg_term = 1.0 - torch.abs(encoded_value)
        loss = mse_loss + self.alpha * reg_term * mse_loss
        print(f'MSE Loss: {mse_loss.item()}, Regularization Term: {reg_term.item()}, Encoded Value: {encoded_value.item()}')
        return loss


# dummy evaluation function
def dummy_evaluate():
    return None

# todo maybe it helps if the GRADIEND mdoel itself (or ModelWithGradiend) knows its interpretaion of encoded values, and thereofre its normalization?
def default_normalization(result, normalization_threshold=0.1):
    if not isinstance(result, dict):
        raise ValueError('Normalization requires a dictionary as evaluate() result!')

    if not len(result) == 3:
        raise ValueError('Default Normalization requires three entries in the result dictionary ("score", mean class 1, mean class 2)!')

    if not 'score' in result:
        raise ValueError('Default Normalization requires a "score" entry in the result dictionary (though not used for normalization, but this indicates a misuse of default_normalization()!')

    keys = list(sorted(result.keys())) # sorting ensures that we always have the same order of keys
    keys = [k for k in keys if k != 'score']  # remove score from keyy
    key1 = keys[0]
    key2 = keys[1]

    if result[key1] < result[key2] and abs(result[key1]) > normalization_threshold and abs(result[key2]) > normalization_threshold:
        print(f'Normalizing GRADIEND because of encoded values: {key1} < {key2}, with values {result[key1]} and {result[key2]}')
        # todo normalize

    return None


deprecated_source_target = {
    'gradient': 'factual',
    'inv_gradient': 'counterfactual',
}

supported_sources_and_targets = set(deprecated_source_target.values()) | {'diff'}

def train(model_with_gradiend,
             output='results/models/model_with_gradiend',
             *,

             # configured special behavior
             #normalization=default_normalization,
             evaluate=dummy_evaluate,

             # training config
             source='factual',
             target='diff',
             criterion=nn.MSELoss(),
             epochs=1,
             batch_size=32,
             batch_size_data=True,
             n_evaluation=250,
             eval_max_size=None,
             do_eval=True,
             lr=1e-5,
             weight_decay=1e-2,
             eps=1e-8,
             max_iterations=None,
             supervised=False,

             # other config
             keep_only_best=True,
             delete_models=False,
             checkpoints=False,
             torch_dtype=torch.float32,
             use_cached_gradients=False, # not supported yet
             normalize_gradiend=True,
             data=None,
             ):

    if hasattr(model_with_gradiend, 'gradiend') and model_with_gradiend.gradiend is not None:
        print(f'Training a GRADIEND model over {len(model_with_gradiend.gradiend.encoder[0].weight)} neurons')
        print('Output:', output)

    if data is None:
        raise ValueError('No dataloader provided! Please provide a dataloader with the training data.')

    if source in deprecated_source_target:
        print(f'Warning: The source "{source}" is deprecated and will be replaced with "{deprecated_source_target[source]}".')
        source = deprecated_source_target[source]
    assert source in supported_sources_and_targets, f'Unsupported source: {source}. Supported sources are: {supported_sources_and_targets}'

    if target in deprecated_source_target:
        print(f'Warning: The target "{target}" is deprecated and will be replaced with "{deprecated_source_target[target]}".')
        target = deprecated_source_target[target]
    assert target in supported_sources_and_targets, f'Unsupported target: {target}. Supported targets are: {supported_sources_and_targets}'

    is_llama = 'llama' in model_with_gradiend.base_model.name_or_path.lower()

    if supervised:
        # todo if possible, check that n_features is the same for GRADIEND and labels
        if target != None:
            print(f'Setting target to None (currently set to {target}) since supervised training does not require a target.')
            target = None

    # config parameters of the training (these don't change during training)
    training_data_stats = {
        'n': len(data)
    }


    config = {
        'output': output,
        'source': source,
        'target': target,
        'criterion': str(criterion),
        'epochs': epochs,
        'batch_size': batch_size,
        'batch_size_data': batch_size_data,
        'n_evaluation': n_evaluation,
        'eval_max_size': eval_max_size,
        'do_eval': do_eval,
        'lr': lr,
        'weight_decay': weight_decay,
        'eps': eps,
        'max_iterations': max_iterations,
        'activation': str(model_with_gradiend.gradiend.activation),
        'base_model': model_with_gradiend.base_model.name_or_path,
        'layers': model_with_gradiend.gradiend.layers,
        'bias_decoder': model_with_gradiend.gradiend.bias_decoder,
        'keep_only_best': keep_only_best,
        'delete_models': delete_models,
        'checkpoints': checkpoints,
        'torch_dtype': str(torch_dtype),
        'use_cached_gradients': use_cached_gradients,
        'normalize_gradiend': normalize_gradiend,
        'data': training_data_stats,
    }

    if model_with_gradiend.base_model.dtype != torch_dtype:
        model_with_gradiend = model_with_gradiend.to(dtype=torch_dtype)

    last_losses = []
    last_losses2 = []
    losses = []
    losses2 = []
    max_losses = 100
    max_losses2 = 1000 # keep track of a 2nd moving average for compatibility reasons

    best_score_checkpoint = None
    global_step = 0

    total_training_time_start = time.time()


    training_stats = {
        'global_step': 0,
        'score': -1.0,
        'scores': {},
        'encoder_norms': [],
        'decoder_norms': [],
    }

    training_stats_excluded_from_normalization = {'global_step', 'encoder_norms', 'decoder_norms'}

    time_stats = {
        'data_preparation': 0.0,
        'model_with_gradiend': 0.0,
        'eval': 0.0,
        'total': time.time() - total_training_time_start
    }

    factual_computation_required_keywords = {'factual', 'diff'}
    counterfactual_computation_required_keywords = {'counterfactual', 'diff'}

    def _evaluate():
        model_with_gradiend.eval()
        eval_result = evaluate(config=config, training_stats=training_stats)


        if eval_result is None and normalize_gradiend:
            raise ValueError('Normalization is only possible if evaluation is enabled!')

        # todo normalization for multi feature gradiends will be more complicated as normalization may require switching of classes

        if 'mean_by_class' in eval_result:
            print('Mean encoded values by class:', eval_result['mean_by_class'])

        if normalize_gradiend:
            if model_with_gradiend.gradiend.latent_dim == 1 and 'mean_by_class' in eval_result:
                if 0 in eval_result['mean_by_class'] and 1 in eval_result['mean_by_class']:

                    try:
                        #class_0_encodings = eval_result['mean_by_class'][0][0]
                        #class_1_encodings = eval_result['mean_by_class'][1][0]
                        score = eval_result['score']
                        if score < -0.5: #class_0_encodings < -0.5 and class_1_encodings > 0.5:
                            #print(f'Invert encoding since class 0 mean encoded value is {class_0_encodings}<-0.5 and class 1 encoded value is {class_1_encodings}>0.5')
                            print(f'Invert encoding since score is {score}<-0.5')
                            model_with_gradiend.invert_encoding()
                            # the eval_result still contains the not normalized values, but this will be corrected after the next evaluation
                    except Exception:
                        pass
                else:
                    print(f'Normalization not possible since no encoded values for class 0 or class 1 found! Encodings: {eval_result["mean_by_class"]}')

            else:
                return eval_result

                raise ValueError('Normalization is only implemented for single feature GRADIENDs yet!')
                # negation of score not possible (there should be a score per class; and the total score is
                for encoded_class, (class_key1, class_key2) in model_with_gradiend.gradiend.encoded_classes.items():
                    assert class_key1 in eval_result and class_key2 in eval_result, f'Encoded class {encoded_class} not found in evaluation result!'

        model_with_gradiend.train()
        return eval_result

    # Training loop
    optimizer_gradiend = torch.optim.AdamW(model_with_gradiend.gradiend.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)

    len_dataloader = len(data)

    if len_dataloader == 0:
        raise ValueError('Dataloader is empty! Please provide a dataloader with training data.')


    for epoch in range(epochs):

        if max_iterations and global_step >= max_iterations:
            print(f'Max iterations {max_iterations} reached, stopping training.')
            break

        dataloader_iterator = tqdm(data, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)

        data_prep_start = time.time()
        for i, batch in enumerate(dataloader_iterator):
            ####### Data Preparation ########
            # the data is already prepared in the dataloader, so we just need to extract the tensors
            source_tensor = batch['source']
            target_tensor = batch['target']

            if supervised:
                labels = torch.tensor(batch['label'])

                if len(labels.shape) > 1:
                    label_dim = labels.shape[1]
                else:
                    if batch_size > 1:
                        label_dim = 1
                    else:
                        label_dim = len(labels)

                gradiend_n_features = model_with_gradiend.gradiend.latent_dim
                if label_dim != gradiend_n_features:
                    raise ValueError(f'Label dimension {label_dim} does not match GRADIEND latent dimension {gradiend_n_features}! Supervised training is only possible with equal feature and label dimensions!.')

            time_stats['data_preparation'] += time.time() - data_prep_start
            ######## END Data Preparation ########

            ######## Gradiend Training ########
            gradiend_start = time.time()

            if source_tensor.device != model_with_gradiend.gradiend.device_encoder:
                source_tensor = source_tensor.to(model_with_gradiend.gradiend.device_encoder)

            if supervised:
                # forward pass only through GRADIEND encoder
                encoded_value = model_with_gradiend.gradiend.encoder(source_tensor)
                del source_tensor
                label = torch.tensor(batch['label'])

                if label.device != encoded_value.device:
                    label = label.to(encoded_value.device)

                if batch_size > 1:
                    label = label.to(torch.float32).mean(dim=0, keepdim=True).to(dtype=encoded_value.dtype)
                else:
                    label = label.to(dtype=encoded_value.dtype).unsqueeze(0)

                loss_gradiend = criterion(encoded_value, label)
            else:
                # Forward pass through GRADIEND
                outputs_gradiend, encoded_value = model_with_gradiend.gradiend(source_tensor, return_encoded=True)
                del source_tensor

                # calculate loss
                if target_tensor.device != outputs_gradiend.device:
                    #target_tensor = target_tensor.to(outputs_gradiend.device)
                    outputs_gradiend = outputs_gradiend.to(target_tensor.device)

                # release memory
                torch.cuda.empty_cache()
                gc.collect()

                if isinstance(criterion, PolarFeatureLoss):
                    loss_gradiend = criterion(outputs_gradiend, target_tensor, encoded_value)
                else:
                    loss_gradiend = criterion(outputs_gradiend, target_tensor)
                del target_tensor


            optimizer_gradiend.zero_grad()
            if is_llama:
                del batch
                del encoded_value
                # free memory
                torch.cuda.empty_cache()
                gc.collect()

            loss_gradiend.backward()
            if is_llama:
                del outputs_gradiend
                torch.cuda.empty_cache()
                gc.collect()

            optimizer_gradiend.step()

            loss_gradiend = loss_gradiend.item()
            time_stats['model_with_gradiend'] += time.time() - gradiend_start
            ###### End Gradiend Training ########

            if len(last_losses) < max_losses:
                last_losses.append(loss_gradiend)
            else:
                last_losses = last_losses[1:] + [loss_gradiend]

            if len(last_losses2) < max_losses2:
                last_losses2.append(loss_gradiend)
            else:
                last_losses2 = last_losses2[1:] + [loss_gradiend]

            training_stats['global_step'] += 1
            global_step = training_stats['global_step']


            last_iteration = global_step == max_iterations or (epoch == epochs - 1 and i == len_dataloader - 1)
            if do_eval and ((i+1) % n_evaluation == 0 or i == 0) or last_iteration:
                eval_start = time.time()
                eval_result = _evaluate()
                if 'main' in eval_result:
                    eval_result = eval_result['main']

                for key, value in eval_result.items():
                    if key not in training_stats:
                        #raise ValueError(f'Key {key} not found in training_stats! Please ensure that the evaluation function returns a dictionary with the expected keys.')
                        # todo this might also just be a warning instead as it is fixable
                        training_stats[key] = {}

                    if isinstance(training_stats[key], dict):
                        training_stats[key][global_step] = value
                    elif isinstance(training_stats[key], list):
                        training_stats[key].append(value)
                    elif isinstance(training_stats[key], float):
                        training_stats[key] = value
                    else:
                        raise ValueError(f'Unsupported type for training_stats[{key}]: {type(training_stats[key])}. Expected dict, list or float.')

                time_stats['eval'] += time.time() - eval_start


            n_loss_report = n_evaluation if n_evaluation > 0 else 100
            if ((i+1) % n_loss_report == 0 or  i == 0) or last_iteration:
                mean_loss = sum(last_losses) / len(last_losses)
                mean_loss2 = sum(last_losses2) / len(last_losses2)
                encoder_norm = model_with_gradiend.gradiend.encoder_norm
                decoder_norm = model_with_gradiend.gradiend.decoder_norm
                avg_grad_norm = model_with_gradiend.gradiend.avg_gradient_norm
                score = training_stats['score']
                output_str = f'Epoch [{epoch + 1}/{epochs}], Loss AE: {mean_loss:.10f}, Correlation score: {score:.6f}, encoder {encoder_norm}, decoder {decoder_norm}, avg grad norm {avg_grad_norm}'

                print(output_str)
                losses.append(mean_loss)
                losses2.append(mean_loss2)

                # update training stats
                training_stats['encoder_norms'].append(encoder_norm)
                training_stats['decoder_norms'].append(decoder_norm)

                if best_score_checkpoint is None or abs(score) >= abs(best_score_checkpoint['score']):
                    if best_score_checkpoint is None:
                        print('First score:', score, 'at global step', global_step)
                        if is_llama:
                            torch.cuda.empty_cache()
                            gc.collect()
                    elif abs(score) > abs(best_score_checkpoint['score']):
                        print('New best score:', score, 'at global step', global_step)
                    else:
                        print('Same score:', score, 'at global step', global_step)

                    best_score_checkpoint = {
                        'score': score,
                        'global_step': global_step,
                        'epoch': epoch,
                        'loss': mean_loss
                    }


                    training_stats['epoch'] = epoch

                    # save checkpoint
                    training_information = {
                        'losses': losses,
                        'losses_1000': losses2,
                        'best_score_checkpoint': best_score_checkpoint,
                        'n_evaluation': n_evaluation,
                        'global_step':  global_step,
                        'eval_max_size': eval_max_size,
                        'time': time_stats,
                        'stats': training_stats,
                        'config': config,
                    }
                    if i > 1:
                        model_with_gradiend.save_pretrained(f'{output}_best', training=training_information)


            training_stats['global_step'] = global_step

            if max_iterations and global_step >= max_iterations:
                break

            if i > 0 and checkpoints:
                checkpoint_step = 5000 if checkpoints is True else checkpoints
                if global_step % checkpoint_step == 0:
                    model_name = f'{output}_{global_step}'
                    model_with_gradiend.save_pretrained(model_name)
                    print('Saved intermediate result')

            #if is_llama:
            #    torch.cuda.empty_cache()
            #    gc.collect()

            # restart the data preparation timer
            data_prep_start = time.time()

        training_information = {
            'losses': losses, # todo
            'best_score_checkpoint': best_score_checkpoint,
            'n_evaluation': n_evaluation,
            'global_step': global_step,
            'eval_max_size': eval_max_size,
            'time': time_stats,
            'stats': training_stats,
            'config': config,
        }

        model_with_gradiend.gradiend.save_pretrained(output, training=training_information)
        print('Saved the GRADIEND model as', output)
        print('Best score:', best_score_checkpoint)
        if epochs > 1 and not (keep_only_best or delete_models):
            output_epoch = f'{output}_epoch_{epoch + 1}'
            model_with_gradiend.save_pretrained(output_epoch, training=training_information)

        try:
            import humanize
            import datetime

            def humanize_time(seconds):
                return humanize.naturaldelta(datetime.timedelta(seconds=seconds))
            print(f'Epoch {epoch + 1}/{epochs} finished')
            print('Total Training time:', humanize_time(training_information['time']['total']))
            print('Training data preparation time:', humanize_time(training_information['time']['data_preparation']))
            print('Training Evaluation time:', humanize_time(training_information['time']['eval']))
            print('Training GRADIEND time:', humanize_time(training_information['time']['model_with_gradiend']))
        except ModuleNotFoundError:
            print('Please install humanize to get a human-readable training time')

    print('Best score:', best_score_checkpoint)

    # release memory
    del model_with_gradiend

    # Call garbage collector
    gc.collect()

    # Empty the CUDA cache
    torch.cuda.empty_cache()

    if keep_only_best:
        # delete the output folder
        output_temp = f'{output}_temp'
        os.rename(output, output_temp)
        # rename the output_best folder to output
        os.rename(f'{output}_best', output)

        # copy all *.pdf and *.png files from the temp folder to the output folder
        for file in os.listdir(output_temp):
            if file.endswith('.pdf') or file.endswith('.png'):
                shutil.copy(os.path.join(output_temp, file), output)

        # copy all subfolders from the temp folder to the output folder
        for folder in os.listdir(output_temp):
            if os.path.isdir(os.path.join(output_temp, folder)):
                shutil.copytree(os.path.join(output_temp, folder), os.path.join(output, folder))

        shutil.rmtree(output_temp)

    if delete_models:
        # delete all *.bin files in the output folder
        for file in os.listdir(output):
            if file.endswith('.bin'):
                os.remove(os.path.join(output, file))

    print('Saved the GRADIEND model as', output)
    return output

def call_train_v2(
        setup,
        model_with_gradiend,
        batch_size_data=True,
        normalize_gradiend=True,
        source='factual',
        target='diff',
        batch_size=32,
        eval_batch_size=32,
        eval_max_size=None,
        torch_dtype=torch.float32,
        #use_gradients=True,
        **kwargs
        ):
    tokenizer = model_with_gradiend.tokenizer
    if batch_size_data is True:
        batch_size_data = batch_size

    is_generative = model_with_gradiend.is_generative

    dataset = setup.create_training_data(tokenizer,
                                      max_size=None,
                                      split='train',
                                      batch_size=batch_size_data,
                                      is_generative=is_generative,
                                      )
    batch_size = dataset.batch_size # the batch_size could have been changed by the setup.create_training_data method (e.g., by using parameter combination that disallows batch_size>1)

    gradient_data = ModelFeatureTrainingDataset(dataset,
                                                tokenizer,
                                                model_with_gradiend.feature_creator,
                                                model_with_gradiend.feature_creator_id,
                                                source=source,
                                                target=target,
                                                dtype=torch_dtype,
                                                device=model_with_gradiend.gradiend.device_encoder,
                                                )


    # we use a GRADIEND batch size of 1, the batch_size parameter of call_train is the one to average gradiend gradients over as inputs
    dataloader = DataLoader(gradient_data, batch_size=1, shuffle=False)

    if eval_max_size == 0:
        def evaluate(*args, **kwargs):
            return {}
    else:
        eval_data = setup.create_eval_data(model_with_gradiend,
                                        split='val',
                                        source=source,
                                        max_size=eval_max_size,
                                        is_generative=is_generative,
                                        pre_load_gradients=False,
                                        )

        evaluate = partial(setup.evaluate, model_with_gradiend, eval_data, eval_batch_size=eval_batch_size)

    config = {
        'batch_size_data': batch_size_data,
    }
    train(model_with_gradiend,
             data=dataloader,
             evaluate=evaluate,
             source=source,
             target=target,
             batch_size=batch_size,
             normalize_gradiend=normalize_gradiend,
             torch_dtype=torch_dtype,
             **config,
             **kwargs
             )


    setup.post_training(model_with_gradiend)

def create_model_with_gradiend(model,
                               layers=None,
                               activation='tanh',
                               activation_decoder='id',
                               bias_decoder=True,
                               grad_iterations=1,
                               decoder_factor=1.0,
                               seed=0,
                               torch_dtype=torch.float32,
                               latent_dim=1,
                               use_gradients=True,
                               post_processing=None,
                               #use_decoder_only_mlm_head=False,
                               setup=None,
                               **kwargs
                               ):
    from gradiend.model import ModelWithGradiend

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    kwargs['torch_dtype'] = torch_dtype
    model = ModelWithGradiend.from_pretrained(model,
                                             layers,
                                             activation=activation,
                                             activation_decoder=activation_decoder,
                                             bias_decoder=bias_decoder,
                                             grad_iterations=grad_iterations,
                                             decoder_factor=decoder_factor,
                                             torch_dtype=torch_dtype,
                                             latent_dim=latent_dim,
                                             use_gradients=use_gradients,
                                             init_gradiends=setup.init_gradiends,
                                             )

    if post_processing is not None:
        model = post_processing(model)

    return model, kwargs

train_method = call_train_v2

def train_single_layer_gradiend(setup, model, layer='base_model.encoder.layer.10.output.dense.weight', **kwargs):
    model_with_gradiend, kwargs = create_model_with_gradiend(model, [layer], latent_dim=setup.n_features, setup=setup, **kwargs)
    return train_method(setup, model_with_gradiend, **kwargs)

def train_multiple_layers_gradiend(setup, model, layers, **kwargs):
    model_with_gradiend, kwargs = create_model_with_gradiend(model, layers, latent_dim=setup.n_features, setup=setup, **kwargs)
    return train_method(setup, model_with_gradiend, **kwargs)

def train_all_layers_gradiend(setup, model='bert-base-cased', **kwargs):
    model_with_gradiend, kwargs = create_model_with_gradiend(model, latent_dim=setup.n_features, setup=setup, **kwargs)
    return train_method(setup, model_with_gradiend, **kwargs)



if __name__ == '__main__':
    train_all_layers_gradiend('bert-base-cased',
                              output='results/models/model_with_gradiend',
                              checkpoints=False,
                              max_iterations=1000000,
                              criterion=nn.MSELoss(),
                              batch_size=8,
                              batch_size_data=None,
                              activation='relu',
                              )
