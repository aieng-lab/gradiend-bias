import copy
import gc
import re
import warnings
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
from torch.nn import Parameter
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig, GPT2Config, \
    AutoModelForSequenceClassification
import json
import os

from transformers.modeling_outputs import MaskedLMOutput

from gradiend.training.decoder_only_mlm.model import DecoderModelWithMLMHead
from gradiend.util import hash_it, convert_tuple_keys_recursively

HF_TOKEN = os.getenv('HF_TOKEN')


class AutoModelForLM(nn.Module):

    @classmethod
    def from_pretrained(self, name_or_path, torch_dtype=torch.float32):
        try:
            return AutoModelForMaskedLM.from_pretrained(name_or_path)
        except Exception:

            config_file = os.path.join(name_or_path, 'config_mlm_head.json')

            if os.path.exists(config_file):
                return DecoderModelWithMLMHead.from_pretrained(name_or_path, torch_dtype=torch_dtype)
            elif 'llama' in name_or_path.lower():
                return AutoModelForCausalLM.from_pretrained(name_or_path,
                                                            torch_dtype=torch_dtype,
                                                            token=HF_TOKEN,
                                                            )

            return AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch_dtype)




class InstructTokenizerWrapper:
    system_prompt_mlm = """
    You are a language model that fills in masked words. In the following sentence, all [MASK] tokens refer to the same word. 
    Your task is to predict the missing word and return only that word — no explanation, no formatting, nothing else.
    """

    system_prompt = """
    You are a language model that completes sentences. Predict the next word that naturally follows the given text. 
    Return only that word — no punctuation, no quotes, and no explanations.
    """

    system_prompt_name = """
You are a language model trained to predict first names. In the following text, [NAME] represents a placeholder for a 
first name. Your task is to predict the most likely name that fits the context. Return only the predicted name — no 
punctuation, no quotation marks, and no explanations.
    """

    def __init__(self, tokenizer, user_prompt_header="user", assistant_prompt_header="assistant"):
        self.tokenizer = tokenizer
        self.user_prompt_header = user_prompt_header
        self.assistant_prompt_header = assistant_prompt_header

        # You can change these markers depending on the model
        self.BEGIN = "<|begin_of_text|>"
        self.START = "<|start_header_id|>"
        self.END = "<|end_header_id|>"
        self.EOT = "<|eot_id|>"

    def _wrap_prompt(self, user_text):
        if isinstance(user_text, str):
            user_texts = [user_text]
        elif isinstance(user_text, list):
            user_texts = user_text
        else:
            raise TypeError("user_text must be a string or a list of strings")

        prompts = []

        for text in user_texts:
            parts = [self.BEGIN]

            # Optional: add a system prompt first
            if self.system_prompt:
                parts.append(f"{self.START}system{self.END}\n{self.system_prompt}\n{self.EOT}")

            # Add user prompt
            parts.append(f"{self.START}{self.user_prompt_header}{self.END}\n{text}\n{self.EOT}")

            # Indicate the assistant is expected to reply
            parts.append(f"{self.START}{self.assistant_prompt_header}{self.END}\n")

            prompts.append(''.join(parts))

        return prompts if len(prompts) > 1 else prompts[0]

    def __call__(self, text, **kwargs):
        """
        Fully mimic Hugging Face tokenizer call: return dict with 'input_ids', 'attention_mask', etc.
        """
        if 'add_special_tokens' in kwargs and not kwargs['add_special_tokens']:
            # If add_special_tokens is False, we don't need to wrap the prompt
            wrapped = text
        else:
            wrapped = self._wrap_prompt(text)

        # our implementation adds special tokens by wrapping the prompt
        kwargs['add_special_tokens'] = False

        return self.tokenizer(wrapped, **kwargs)

    def tokenize(self, text, **kwargs):
        wrapped = self._wrap_prompt(text)
        return self.tokenizer.tokenize(wrapped, **kwargs)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def __getattr__(self, name):
        # Fallback to base tokenizer for any other attributes/methods
        return getattr(self.tokenizer, name)

    def __setattr__(self, key, value):
        if key in ['tokenizer', 'system_prompt']:
            super().__setattr__(key, value)
        else:
            setattr(self.tokenizer, key, value)

class AutoTokenizerForLM(AutoTokenizer):
    @classmethod
    def from_pretrained(cls, name, *args, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(name, token=HF_TOKEN, *args, **kwargs)

        if "instruct" in name.lower():
            return InstructTokenizerWrapper(tokenizer)
        else:
            return tokenizer


def freeze_layers_until_target(model, *target_layer_names):
    assert len(target_layer_names) > 0, 'At least one target layer name must be provided'

    # iterate through the parameters from the model, starting at the input layer
    # we stop iterating as soon as we hit the first target layer
    for name, param in model.named_parameters():
        if name in target_layer_names:
            return

        param.requires_grad = False



def get_activation(activation: str, encoder=False):
    if activation == 'relu':
        activation_fnc = nn.ReLU(inplace=True)
    elif activation == 'leakyrelu':
        activation_fnc = nn.LeakyReLU(inplace=True)
    elif activation == 'tanh':
        activation_fnc = nn.Tanh()
    elif activation == 'smht':
        activation_fnc = nn.Hardtanh()
    elif activation == 'elu':
        activation_fnc = nn.ELU()
    elif activation == 'gelu':
        activation_fnc = nn.GELU()
    elif activation == 'sigmoid':
        activation_fnc = nn.Sigmoid()
    elif activation == 'id':
        if encoder:
            activation_fnc = nn.LayerNorm(1)
        else:
            activation_fnc = nn.Identity()
    elif activation == 'silu':
        activation_fnc = nn.SiLU()
    else:
        raise ValueError('Unsupported activation function:', activation)
    return activation_fnc

class LargeLinear(nn.Module):
    """
        A linear layer that handles both standard and very large input feature sizes
        by using chunked computation when the input dimension exceeds the limit
        for standard CUDA BLAS GEMM operations (approximately 2.14 billion).

        Internally, it uses a standard `torch.nn.Linear` layer for parameter
        management and efficient computation for smaller inputs.

Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        in_chunk_size (int, optional): The size of the chunks to process the
            input dimension in when it exceeds the standard limit.
            Default: 2,000,000,000.
        out_chunk_size (int, optional): The size of the chunks to process the
            output dimension in when it exceeds the standard limit.
            Default: 2,000,000,000.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 in_chunk_size=2000000000,
                 out_chunk_size=2000000000,
                 dtype=torch.float32,
                 device=None,
                 ):
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.in_chunk_size = in_chunk_size
        self.out_chunk_size = out_chunk_size
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, input):
        if input.device != self.linear.weight.device:
            input = input.to(self.linear.weight.device)

        input_size = input.size(-1)
        output_size = self.out_features
        max_size = np.iinfo(np.int32).max
        if input_size <= max_size and output_size <= max_size:  # Standard computation if within limit
            return self.linear(input)
        elif output_size > max_size:
            # Chunking along the output dimension
            num_out_chunks = (output_size + self.out_chunk_size - 1) // self.out_chunk_size

            # determine output shape
            output_shape = list(input.shape[:-1]) + [output_size]
            output = torch.zeros(*output_shape, device=input.device, dtype=input.dtype)

            for i in range(num_out_chunks):
                out_start = i * self.out_chunk_size
                out_end = min((i + 1) * self.out_chunk_size, output_size)
                weight_chunk = self.linear.weight[out_start:out_end, :].to(input.device)
                bias_chunk = self.linear.bias[out_start:out_end] if self.linear.bias is not None else None

                if input_size > max_size:
                    # Also chunk along the input dimension
                    # todo test this if-branch!
                    num_in_chunks = (input_size + self.in_chunk_size - 1) // self.in_chunk_size
                    intermediate_parts = []
                    for j in range(num_in_chunks):
                        in_start = j * self.in_chunk_size
                        in_end = min((j + 1) * self.in_chunk_size, input_size)
                        input_chunk = input[..., in_start:in_end]
                        weight_in_chunk = weight_chunk[:, in_start:in_end]
                        output_in_chunk = torch.matmul(input_chunk.unsqueeze(-2),
                                                       weight_in_chunk.T.unsqueeze(-1)).squeeze(-1)
                        intermediate_parts.append(output_in_chunk)
                    output_part = torch.sum(torch.stack(intermediate_parts, dim=-1), dim=-1)
                else:
                    output_part = torch.matmul(input, weight_chunk.T).squeeze(-1)


                if bias_chunk is not None:
                    output_part += bias_chunk.to(input.device)

                output[..., out_start:out_end] = output_part

            return output
        elif input_size > max_size:
            # Chunked computation
            num_chunks = (input_size + self.in_chunk_size - 1) // self.in_chunk_size
            outputs = []
            for i in range(num_chunks):
                start = i * self.in_chunk_size
                end = min((i + 1) * self.in_chunk_size, input_size)
                input_chunk = input[..., start:end]
                weight_chunk = self.linear.weight[:, start:end].to(input.device)
                output_chunk = F.linear(input_chunk, weight_chunk, None) # Bias is added once at the end
                outputs.append(output_chunk)

            output = torch.sum(torch.stack(outputs, dim=-1), dim=-1)
            bias = self.linear.bias
            if bias is not None:
                output += bias.to(input.device)
            return output
        else:
            raise ValueError()

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        if not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value)
        self.linear.weight = value  # Avoid re-registering

    @property
    def bias(self):
        return self.linear.bias

    @bias.setter
    def bias(self, value):
        if value is not None and not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value)
        self.linear.bias = value  # Avoid re-registering




class ModelForClassificationWithMLM(nn.Module):
    """
    Acts as a BertForMaskedLM during forward,
    but only saves the classification model when calling save_pretrained().
    """

    def __init__(self, cls_checkpoint: str, mlm_checkpoint: str = None):
        super().__init__()
        # Load classification model (base encoder + classifier)

        self.cls_model = AutoModelForSequenceClassification.from_pretrained(cls_checkpoint)

        # find encoder module (works for bert, distilbert, roberta, xlm-roberta, ...)
        for candidate in ("bert", "distilbert", "roberta", "xlm_roberta", "electra", "albert"):
            if hasattr(self.cls_model, candidate):
                self.encoder_name = candidate
                self.encoder = getattr(self.cls_model, candidate)
                break
        else:
            # fallback: some HF wrappers expose .base_model
            if hasattr(self.cls_model, "base_model"):
                self.encoder = self.cls_model.base_model
                self.encoder_name = getattr(self.cls_model, "base_model_prefix", "base_model")
            else:
                raise ValueError("Couldn't locate encoder submodule on classification model")

        # Optionally load MLM head on same base encoder architecture
        if mlm_checkpoint:
            try:
                self.mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_checkpoint)
            except Exception:
                self.mlm_model = AutoModelForCausalLM.from_pretrained(mlm_checkpoint)
            self.mlm_head = self.mlm_model.get_output_embeddings()
        else:
            self.mlm_head = None

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # DistilBERT/others return last_hidden_state as .last_hidden_state or .hidden_states[...]
        sequence_output = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        prediction_scores = self.mlm_head(sequence_output) if self.mlm_head is not None else None

        loss = None
        if labels is not None and prediction_scores is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1),
            )

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    def named_parameters(self, prefix='', recurse=True):
        # expose encoder params + mlm_head params only
        for name, param in self.encoder.named_parameters(prefix=f"{prefix}{self.encoder_name}", recurse=recurse):
            yield name, param
        if self.mlm_head is not None:
            for name, param in self.mlm_head.named_parameters(prefix=f"{prefix}mlm_head", recurse=recurse):
                yield name, param

    def save_pretrained(self, save_directory):
        self.cls_model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, cls_checkpoint: str, mlm_checkpoint: str = None):
        return cls(cls_checkpoint=cls_checkpoint, mlm_checkpoint=mlm_checkpoint)

    @property
    def name_or_path(self):
        return self.cls_model.name_or_path

    @property
    def dtype(self):
        return self.cls_model.dtype



class GradiendModel(nn.Module):
    def __init__(self, input_dim,
                 latent_dim,
                 layers,
                 activation='tanh',
                 bias_decoder=True,
                 decoder_factor=1.0,
                 activation_decoder='id',
                 torch_dtype=torch.float32,
                 device=None,
                 device_encoder=None,
                 device_decoder=None,
                 **kwargs):
        super(GradiendModel, self).__init__()
        self.device_encoder = device_encoder or device or torch.device('cuda')
        self.device_decoder = device_decoder or device or torch.device('cuda')

        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)

        if self.input_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("Input and latent dimensions must be positive integers.")

        self.layers = layers
        self.activation = activation.lower()
        self.bias_decoder = bias_decoder
        self.torch_dtype = torch_dtype
        self.kwargs = kwargs
        if 'base_model' in kwargs and hasattr(kwargs['base_model'], 'name_or_path'):
                self.kwargs['base_model'] = kwargs['base_model'].name_or_path

        activation_fnc = get_activation(self.activation, encoder=True)

        if activation_decoder:
            activation_fnc_decoder = get_activation(activation_decoder)
            self.activation_decoder = activation_decoder
        else:
            activation_fnc_decoder = get_activation(self.activation, encoder=False)
            self.activation_decoder = self.activation

        self.encoder = nn.Sequential(
            LargeLinear(input_dim, latent_dim, dtype=torch_dtype, device=self.device_encoder),
            activation_fnc
        )
        self.decoder = nn.Sequential(
            LargeLinear(latent_dim, input_dim, bias=bias_decoder, dtype=torch_dtype, device=self.device_decoder),
            activation_fnc_decoder
        )

        # Initialize the decoder weights with the same distribution as the encoder weights (up to decoder_factor)
        x = self.encoder[0].weight.max().item() * decoder_factor
        nn.init.uniform_(self.decoder[0].weight, -x, x)

        if bias_decoder:
            nn.init.uniform_(self.decoder[0].bias, -x, x)

        self.avg_gradient_norm = 0.0
        self.ctr = 0

    @property
    def decoder_norm(self):
        return torch.norm(self.decoder[0].weight, p=2).item()

    @property
    def encoder_norm(self):
        return torch.norm(self.encoder[0].weight, p=2).item()

    @property
    def layers_hash(self):
        if isinstance(self.layers, dict):
            layers_keys_hash = hash_it(list(self.layers.keys()))
            sparse_layers = torch.concat([self.layers[k].flatten() for k in self.layers], dim=0).cpu().to_sparse()
            layers_indices = sparse_layers.indices().cpu().numpy()
            layers_values = sparse_layers.values().cpu().numpy()
            layers_hash = hash_it((layers_keys_hash, layers_indices, layers_values))
        else:
            layers_hash = hash_it(self.layers)
        return layers_hash

    def extract_gradients_(self, bert, return_dict=False, ):
        layer_map = {k: v for k, v in bert.named_parameters()}
        if isinstance(self.layers, dict):
            if return_dict:
                return {k: v.grad.detach() * self.layers[k] if v.grad is not None else torch.zeros_like(v) for k, v in layer_map.items() if k in self.layers}
            else:
                layer_map = {k: v.grad[self.layers[k]] for k, v in layer_map.items() if k in self.layers}
            return torch.concat([layer_map[layer].flatten().detach() for layer in self.layers])
        elif return_dict:
            return {layer: layer_map[layer].grad.detach() for layer in self.layers}

        return torch.concat([layer_map[layer].grad.flatten().detach() for layer in self.layers])

    def extract_gradients(self, bert, return_dict=False):
        layer_map = {k: v for k, v in bert.named_parameters()}
        if isinstance(self.layers, dict):
            if return_dict:
                return {
                    k: (v.grad.detach().clone() * self.layers[k]
                        if v.grad is not None else torch.zeros_like(v))
                    for k, v in layer_map.items() if k in self.layers
                }
            else:
                layer_map = {
                    k: v.grad[self.layers[k]].detach().clone()
                    for k, v in layer_map.items() if k in self.layers
                }
                return torch.concat([layer_map[layer].flatten() for layer in self.layers])
        elif return_dict:
            return {layer: layer_map[layer].grad.detach().clone()
                    for layer in self.layers}

        return torch.concat([layer_map[layer].grad.flatten().detach().clone()
                             for layer in self.layers])

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self, x, return_encoded=False):
        orig_shapes = {}

        if hasattr(x, 'named_parameters'):
            grads = []
            layer_map = {k: v for k, v in x.named_parameters()}
            if isinstance(self.layers, dict):
                # New behavior: process only masked elements
                for layer, mask in self.layers.items():
                    param = layer_map[layer]
                    grad = param.grad.flatten()
                    selected_grad = grad[mask.flatten()]  # Extract relevant elements
                    grads.append(selected_grad)
                    orig_shapes[layer] = (param.shape, mask)  # Store shape & mask
            else:
                for layer in self.layers:
                    param = layer_map[layer]
                    grad = param.grad.flatten()
                    grads.append(grad)
                    orig_shapes[layer] = param.shape
            x = torch.concat(grads)
        elif isinstance(x, dict):
            grads = []
            if isinstance(self.layers, dict):
                for layer, mask in self.layers.items():
                    param = x[layer]
                    grad = param.flatten()
                    selected_grad = grad[mask.flatten()]  # Extract relevant elements
                    grads.append(selected_grad)
                    orig_shapes[layer] = (param.shape, mask)  # Store shape & mask
            else:
                for layer in self.layers:
                    param = x[layer]
                    grad = param.flatten()
                    grads.append(grad)
                    orig_shapes[layer] = param.shape
            x = torch.concat(grads)

        encoded = self.encoder(x)
        if encoded.device != self.device_decoder:
            encoded = encoded.to(self.device_decoder)
        decoded = self.decoder(encoded)

        grad_norm = torch.norm(x, p=2).item()
        self.avg_gradient_norm = (self.avg_gradient_norm * self.ctr + grad_norm) / (self.ctr + 1)
        self.ctr += 1

        if orig_shapes:
            decoded_params = {}
            start_idx = 0
            for layer, shape_info in orig_shapes.items():
                if isinstance(shape_info, tuple):
                    shape, mask = shape_info  # Extract shape and mask
                    num_elements = mask.sum().item()

                    # Reconstruct full tensor, placing decoded values only where mask is True
                    reconstructed = torch.zeros_like(mask, dtype=decoded.dtype)
                    reconstructed[mask] = decoded[start_idx:start_idx + num_elements]
                    decoded_params[layer] = reconstructed.reshape(shape)
                else:
                    shape = shape_info
                    num_elements = shape.numel()
                    decoded_params[layer] = decoded[start_idx:start_idx + num_elements].reshape(shape)

            decoded = decoded_params

        if return_encoded:
            return decoded, encoded
        return decoded

    def forward_encoder(self, x):
        if hasattr(x, 'named_parameters'): # todo remove?
            grads = []
            layer_map = {k: v for k, v in x.named_parameters()}
            for layer in self.layers:
                param = layer_map[layer]
                grad = param.grad.flatten()
                grads.append(grad)

            x = torch.stack(grads)

        encoded = self.encoder(x)
        print('Encoded', encoded)
        return encoded

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        config_path = os.path.join(save_directory, 'config.json')
        layers_path = os.path.join(save_directory, 'layers.pth')

        # Save model state dictionary
        torch.save(self.state_dict(), model_path)

        self.kwargs.update(kwargs)

        # Save sparse tensor layers separately
        if isinstance(self.layers, dict):
            torch.save(self.layers, layers_path)

        # Save model configuration
        config = {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'layers': list(self.layers),
            'activation': self.activation,
            'activation_decoder': self.activation_decoder,
            'bias_decoder': self.bias_decoder,
            **self._serialize_kwargs(),
        }
        if isinstance(self.layers, dict):
            config['layers_path'] = 'layers.pth'

        with open(config_path, 'w') as f:
            json.dump(config, f)

    def _serialize_kwargs(self):
        kwargs = self.kwargs.copy()
        training_kwargs = kwargs['training']['config'].copy()

        if isinstance(training_kwargs['layers'], dict):
            training_kwargs['layers'] = list(training_kwargs['layers'].keys())
            training_kwargs['layers_path'] = 'layers.pth'
        kwargs['training']['config'] = training_kwargs

        kwargs = convert_tuple_keys_recursively(kwargs)

        return kwargs

    @classmethod
    def _load_legacy_state_dict(cls, state_dict):
        warnings.warn(
            "You are using a legacy checkpoint format. Please update your model checkpoints. "
            "This fallback support will be removed in future versions.",
            DeprecationWarning
        )

        # Mapping from old keys to new keys
        key_mapping = {
            "encoder.0.weight": "encoder.0.linear.weight",
            "encoder.0.bias": "encoder.0.linear.bias",
            "decoder.0.weight": "decoder.0.linear.weight",
            "decoder.0.bias": "decoder.0.linear.bias",
        }

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = key_mapping.get(k, k)
            new_state_dict[new_key] = v

        return new_state_dict

    @classmethod
    def from_pretrained(cls, load_directory, device_encoder=None, device_decoder=None):
        model_path = os.path.join(load_directory, 'pytorch_model.bin')
        config_path = os.path.join(load_directory, 'config.json')

        # Load model configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        # check if the file actually belongs to a GRADIEND model
        if 'input_dim' not in config or 'latent_dim' not in config or 'layers' not in config:
            raise FileNotFoundError(f"The directory {load_directory} does not contain a valid GRADIEND model configuration.")


        if 'llama' in load_directory.lower() and device_encoder is None and device_decoder is None:
            # check that two GPUs are available
            if torch.cuda.device_count() < 2:
                raise RuntimeError("Two GPUs are required for GRADIEND Llama models.")

            device_encoder = torch.device("cuda:0")
            device_decoder = torch.device("cuda:1")

        # Instantiate the model
        model = cls(**config, device_encoder=device_encoder, device_decoder=device_decoder)

        # Load model state dictionary
        state_dict = torch.load(model_path, map_location=device_decoder, weights_only=True)

        # Check if the model is a legacy checkpoint
        if 'encoder.0.weight' in state_dict and 'decoder.0.weight' in state_dict:
            state_dict = cls._load_legacy_state_dict(state_dict)

        model.load_state_dict(state_dict)

        model.name_or_path = load_directory

        if 'layers_path' in config:
            layers_path = os.path.join(load_directory, config['layers_path'])
            # Load sparse layers
            try:
                model.layers = torch.load(layers_path)
            except FileNotFoundError:
                print(f"Warning: {layers_path} not found. Using all layers by default. This will be deprecated soon. Please do only specify layers_path in config if a layers_path exists")

        return model

    @property
    def grad_iterations(self):
        return self.kwargs.get('grad_iterations', 1)


    def plot(self, fig=None, bins=50, n=None):
        # Initialize lists to store weights and biases
        encoder_weights = []
        decoder_weights = []
        decoder_bias = []

        # Extract weights and biases from encoder and decoder
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                encoder_weights.append(param.flatten().detach().cpu().numpy())
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                decoder_weights.append(param.flatten().detach().cpu().numpy())
            elif 'bias' in name:
                decoder_bias.append(param.flatten().detach().cpu().numpy())

        # Convert lists to numpy arrays
        encoder_weights = np.concatenate(encoder_weights)
        decoder_weights = np.concatenate(decoder_weights)
        decoder_bias = np.concatenate(decoder_bias)

        # Compute bin edges and aggregate values
        encoder_bin_means, encoder_bin_edges, _ = binned_statistic(
            np.arange(len(encoder_weights)), encoder_weights, statistic='mean', bins=bins)
        decoder_bin_means, decoder_bin_edges, _ = binned_statistic(
            np.arange(len(encoder_weights), len(encoder_weights) + len(decoder_weights)),
            decoder_weights, statistic='mean', bins=bins)

        if self.bias_decoder:
            decoder_bias_bin_means, decoder_bias_bin_edges, _ = binned_statistic(
                np.arange(len(encoder_weights) + len(decoder_weights),
                          len(encoder_weights) + len(decoder_weights) + len(decoder_bias)),
                decoder_bias, statistic='mean', bins=bins)

        # Plotting
        if fig is None:
            fig = plt.figure(figsize=(12, 6))
        else:
            fig.clear()

        # Encoder weights (blue)
        plt.scatter(encoder_bin_edges[:-1], encoder_bin_means, color='blue', label='Encoder Weights')

        # Decoder weights (green)
        plt.scatter(decoder_bin_edges[:-1], decoder_bin_means, color='green', label='Decoder Weights')

        plt.title(f'Weights of Encoder and Decoder (Aggregated) {n if n else ""}')
        plt.xlabel('Parameter Index')
        plt.ylabel('Mean Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return

    @property
    def source(self):
        if 'config' in self.kwargs['training']:
            source = self.kwargs['training']['config']['source']
        else:
            # todo deprecate
            source = self.kwargs['training'].get('source', 'factual')

        if source == 'gradient':
            source = 'factual'
        elif source == 'inv_gradient':
            source = 'counterfactual'

        return source

class CombinedGradiendModel(GradiendModel):

    def _extract_property(self, gradiends, property_name, default=None):
        values = [getattr(g, property_name, default) for g in gradiends]
        if all(v == values[0] for v in values):
            return values[0]
        else:
            raise ValueError(f"All gradiends must have the same {property_name}, but got: {values}")

    def __init__(self, gradiends):
        input_dim = self._extract_property(gradiends, 'input_dim')
        latent_dim = len(gradiends) # todo ensure all have latent_dim 1
        layers = self._extract_property(gradiends, 'layers')
        activation = self._extract_property(gradiends, 'activation')
        bias_decoder = self._extract_property(gradiends, 'bias_decoder')
        decoder_factor = self._extract_property(gradiends, 'decoder_factor', 1.0)
        activation_decoder = self._extract_property(gradiends, 'activation_decoder')
        torch_dtype = self._extract_property(gradiends, 'torch_dtype', torch.float32)

        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            layers=layers,
            activation=activation,
            bias_decoder=bias_decoder,
            decoder_factor=decoder_factor,
            activation_decoder=activation_decoder,
            torch_dtype=torch_dtype,
        )

        for i, gradiend in enumerate(gradiends):
            self.encoder[0].weight.data[i:i+1, :] = gradiend.encoder[0].weight.data
            self.encoder[0].bias.data[i:i+1] = gradiend.encoder[0].bias.data
            self.decoder[0].weight.data[:, i:i+1] = gradiend.decoder[0].weight.data

        self.decoder[0].bias.data = torch.stack([g.decoder[0].bias.data for g in gradiends]).mean(dim=0)


def is_generative(model):
    if isinstance(model, ModelForClassificationWithMLM):
        model = model.mlm_model

    if 'bert' in str(type(model)).lower():
        return False

    return not any(x in str(type(model)) for x in ['ForMaskedLM', 'MLMHead', 'WithMLM'])


class ModelWithGradiend(nn.Module):

    def __init__(self, base_model, gradiend, tokenizer, base_model_device=None, use_gradients=True):
        super().__init__()
        self.base_model = base_model
        self.gradiend = gradiend
        self.tokenizer = tokenizer
        self.grad_iterations = gradiend.grad_iterations
        self.use_gradients = use_gradients

        self.base_model_device = base_model_device or gradiend.encoder[0].linear.weight.device
        self.base_model.to(self.base_model_device)
        self.layer_map = {k: v for k, v in self.base_model.named_parameters()}

        self.is_instruction_model = isinstance(self.tokenizer, InstructTokenizerWrapper)
        self.is_generative = is_generative(self.base_model)
        self._enhancer_mask_cache = {}

        if self.is_generative:
            if self.tokenizer.eos_token is None:
                raise ValueError('Tokenizer must have an eos_token for generative models')
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.gradiend.parameters(recurse=recurse)

    def plot(self, *args, **kwargs):
        # Plotting is handled by the gradiend model
        return self.gradiend.plot(*args, **kwargs)

    @property
    def name(self):
        return os.path.basename(self.gradiend.name_or_path)

    @property
    def source(self):
        return self.gradiend.source

    def create_gradients(self, text, label, return_dict=False, verbose=False):
        item = self.create_inputs(text, label)

        outputs = self.base_model(**item)


        if verbose:
            # print the predicted inputs/labels
            labels = item['labels']
            input_ids = item['input_ids']
            #label_idx = labels[labels != -100]
            if self.is_generative:
                logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
                predictions = logits.argmax(dim=-1)  # shape: [batch_size, seq_len]

                # Loop through batch
                for i in range(input_ids.size(0)):
                    label_mask = labels[i] != -100
                    label_positions = label_mask.nonzero(as_tuple=True)[0]

                    print(f"\n[Sample {i}]")
                    for pos in label_positions:
                        pred_id = predictions[i, pos-1].item()
                        true_id = labels[i, pos].item()

                        pred_token = self.tokenizer.decode([pred_id], skip_special_tokens=True)
                        true_token = self.tokenizer.decode([true_id], skip_special_tokens=True)

                        print(f"Position {pos.item():>2}: Predicted = '{pred_token}', True = '{true_token}'",
                              "✅" if pred_token.strip().lower() == true_token.strip().lower() else "❌")

        loss_base_model = outputs.loss

        self.base_model.zero_grad()
        loss_base_model.backward()

        gradients = self.gradiend.extract_gradients(self.base_model, return_dict=return_dict)
        self.base_model.zero_grad(set_to_none=True)
        return gradients

    def create_base_model_outputs(self, text, label):
        # create inputs for the base model
        inputs = self.create_inputs(text, label)
        outputs = self.create_model_embeddings(inputs)
        return outputs

    def create_encoder_inputs(self, text, label):
        if self.use_gradients:
            return self.create_gradients(text, label)
        else:
            return self.create_base_model_outputs(text, label)

    def encode(self, input, label=None, top_k=None, top_k_part=None):
        top_k_part = top_k_part or 'encoder'

        if isinstance(input, str):
            assert label is not None, 'Label must be provided if input is a string'
            input = self.create_encoder_inputs(input, label)

        if top_k is not None:
            # only use the top k elements of the encoder (wrt. to absolute value)
            mask = self.get_enhancer_mask(top_k, part=top_k_part)
            masked_encoder = self.gradiend.encoder[0].weight.flatten()[mask]
            masked_input = input[mask]
            encoded = torch.matmul(masked_input.unsqueeze(0), masked_encoder.unsqueeze(1)).squeeze(1) + self.gradiend.encoder[0].bias
            encoded = self.gradiend.encoder[1](encoded)
        else:
            encoded = self.gradiend.encoder(input)

        return encoded

    def modify_model(self, lr, feature_factor, part='decoder', top_k=None, top_k_part=None):
        # returns a base_model model with enhanced weights based on the GRADIEND, use the learning rate parameter to control the influence of the GRADIEND
        top_k_part = top_k_part or part

        if not isinstance(feature_factor, list):
            feature_factor = [feature_factor]


        import copy
        enhanced_model = copy.deepcopy(self.base_model)

        if top_k == 0:
            return enhanced_model

        model_device = self.base_model.device
        layer_map = {k: v for k, v in enhanced_model.named_parameters()}
        if part == 'decoder':
            enhancer = self.gradiend.decoder(torch.tensor(feature_factor, dtype=torch.float, device=model_device))
        elif part == 'encoder':
            enhancer = self.gradiend.encoder[0].weight.flatten().to(model_device)
        else:
            raise ValueError('Unsupported part:', part)

        if top_k is not None and top_k < len(enhancer):
            mask = self.get_enhancer_mask(top_k, part=top_k_part)

            if mask.sum() == 0.0:
                return enhanced_model

            enhancer[~mask] = 0.0

        idx = 0
        with torch.no_grad():
            if isinstance(self.gradiend.layers, dict):
                for layer, layer_mask in self.gradiend.layers.items():
                    number_of_elements = layer_mask.sum().item()  # Convert to Python integer if it's a tensor

                    # Extract the relevant elements from enhancer
                    update_values = enhancer[idx:idx + number_of_elements].to(model_device)

                    # Create an update tensor of the same shape as layer_mask, filling only True positions
                    update_tensor = torch.zeros_like(layer_mask, dtype=update_values.dtype)
                    update_tensor[layer_mask] = update_values  # Assign values only to True positions

                    # Apply update
                    layer_map[layer] += lr * update_tensor

                    # Increment index
                    idx += number_of_elements
            else:
                for layer in self.gradiend.layers:
                    shape = layer_map[layer].shape
                    number_of_elements = shape.numel()
                    enhancer = enhancer.unsqueeze(0) if enhancer.dim() == 1 else enhancer
                    layer_chunk = enhancer[:, idx:idx + number_of_elements].to(model_device)
                    layer_map[layer] += lr * layer_chunk.reshape(shape)
                    idx += number_of_elements

        return enhanced_model

    def get_enhancer(self, part='decoder'):
        # we set parts of the enhancer to 0 that are not in the top k highest values (wrt absolute value)
        if part == 'decoder':
            abs_enhancer = self.gradiend.decoder[0].weight.flatten().abs()
        elif part == 'decoder-bias':
            abs_enhancer = self.gradiend.decoder[0].bias.abs()
        elif part == 'decoder-sum':
            abs_enhancer = (self.gradiend.decoder[0].weight.flatten() + self.gradiend.decoder[0].bias).abs()
        elif part == 'encoder':
            abs_enhancer = self.gradiend.encoder[0].weight.flatten().abs()
        else:
            raise ValueError('Unsupported part:', part)

        return abs_enhancer


    def get_enhancer_mask(self, top_k, part='decoder'):
        cache_key = f'{top_k}_{part}'
        if cache_key in self._enhancer_mask_cache:
            return self._enhancer_mask_cache[cache_key]

        gradiend_vector = self.gradiend.decoder[0].weight.flatten()

        if top_k == 0.0:
            mask = torch.zeros_like(gradiend_vector, dtype=torch.bool)
            self._enhancer_mask_cache[cache_key] = mask
            return mask

        if 0.0 < top_k <= 1.0:
            top_k = int(top_k * len(gradiend_vector))

        abs_enhancer = self.get_enhancer(part=part).cpu() # move to cpu to ensure deterministic behavior
        sorted_indices = torch.argsort(abs_enhancer, stable=True)  # Ensure stable order
        sorted_enhancer = abs_enhancer[sorted_indices]

        top_k_values, top_k_sorted_indices = sorted_enhancer.topk(top_k, sorted=False, largest=True)

        # Convert back to original indices
        top_k_indices = sorted_indices[top_k_sorted_indices]
        mask = torch.zeros_like(gradiend_vector, dtype=torch.bool)
        mask[top_k_indices] = True
        self._enhancer_mask_cache[cache_key] = mask
        return mask

    def get_layer_mask(self, top_k, part='decoder'):
        enhancer_mask = self.get_enhancer_mask(top_k=top_k, part=part)
        all_layer_map = {k: v for k, v in self.base_model.named_parameters()}
        layer_map = {}

        idx = 0
        with torch.no_grad():
            if isinstance(self.gradiend.layers, dict):
                raise NotImplementedError()
                for layer, layer_mask in self.gradiend.layers.items():
                    shape = layer_map[layer].shape
                    number_of_elements = layer_mask.sum().item()  # Convert to Python integer if it's a tensor

                    # Extract the relevant elements from enhancer
                    #update_values = enhancer[idx:idx + number_of_elements]

                    # Create an update tensor of the same shape as layer_mask, filling only True positions
                    update_tensor = torch.zeros_like(layer_mask, dtype=update_values.dtype)
                    update_tensor[layer_mask] = update_values  # Assign values only to True positions

                    # Apply update
                    layer_map[layer] = lr * update_tensor

                    # Increment index
                    idx += number_of_elements
            else:
                for layer in self.gradiend.layers:
                    shape = all_layer_map[layer].shape
                    number_of_elements = shape.numel()
                    layer_map[layer] = enhancer_mask[idx:idx + number_of_elements].reshape(shape) #.to_sparse()
                    idx += number_of_elements

        return layer_map



    def mask_and_encode(self, text, ignore_tokens=False, return_masked_text=False, single_mask=True, top_k=None, top_k_part=None):
        item = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        item = {k: v.to(self.base_model_device) for k, v in item.items()}
        labels = item['input_ids'].clone()

        if self.is_generative:
            n = labels.shape[1]

            # left shift the labels by one
            labels = torch.cat([labels[:, 1:], torch.full_like(labels[:, :1], self.tokenizer.pad_token_id)], dim=1)

            # use random idx to predict in the 2nd half of the sequence to ensure enough context
            if ignore_tokens:
                # Create a mask for valid indices in the second half, excluding the ignore tokens
                valid_indices_mask = ~torch.isin(labels[0, n // 2:], torch.tensor(ignore_tokens, device=labels.device))

                # Get the indices of the valid tokens
                valid_indices = torch.nonzero(valid_indices_mask, as_tuple=False).squeeze(-1)

                if valid_indices.numel() > 0:
                    # Randomly select an index from the valid tokens
                    random_idx = valid_indices[torch.randint(0, valid_indices.numel(), (1,))] + n // 2
                else:
                    # Handle case where no valid indices exist (fallback behavior, could raise an error or select any index)
                    raise ValueError('No valid indices found in the second half of the sequence', text)
            else:
                random_idx = torch.randint(n // 2, n, (1,))

            labels[:, :random_idx-1] = self.tokenizer.pad_token_id
            labels[:, random_idx:] = self.tokenizer.pad_token_id

            mask = labels != self.tokenizer.pad_token_id
        else:
            if single_mask:
                mask = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
                # randomly mask a single entry
                mask[0, torch.randint(0, labels.shape[1], (1,))] = True
            else:
                random_mask = torch.rand(labels.shape, dtype=torch.float, device=labels.device) < 0.15
                padding_mask = labels == self.tokenizer.pad_token_id

                if ignore_tokens:
                    exclude_mask = (labels.unsqueeze(-1) == torch.Tensor(ignore_tokens).to(labels.device)).any(dim=-1)
                else:
                    exclude_mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
                mask = random_mask & ~padding_mask & ~exclude_mask
            labels[~mask] = -100  # only compute loss on masked tokens

            item['input_ids'][mask] = self.tokenizer.mask_token_id

        item['labels'] = labels

        outputs = self.base_model(**item)
        loss_base_model = outputs.loss
        if loss_base_model is None:
            if return_masked_text:
                return None, None, None
            return None

        self.base_model.zero_grad()
        loss_base_model.backward()

        gradients = self.gradiend.extract_gradients(self.base_model)
        self.base_model.zero_grad(set_to_none=True)

        if top_k is not None:
            top_k_part = top_k_part or 'encoder'
            enhancer_mask = self.get_enhancer_mask(top_k=top_k, part=top_k_part)
            masked_encoder = self.gradiend.encoder[0].weight.flatten()[enhancer_mask]
            masked_input = gradients[enhancer_mask]
            encoded = torch.matmul(masked_input.unsqueeze(0), masked_encoder.unsqueeze(1)).squeeze(1) + self.gradiend.encoder[0].bias
            encoded = self.gradiend.encoder[1](encoded)
        else:
            encoded = self.gradiend.encoder(gradients)

        if hasattr(encoded, 'tolist'):
            encoded = encoded.tolist()

        if len(encoded) == 1:
            encoded = encoded[0]

        if return_masked_text:
            masked_str = self.tokenizer.decode(item['input_ids'].squeeze())
            masked_str = masked_str.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()
            labels = [self.tokenizer.decode(id) for id in item['labels'][mask]]
            return encoded, masked_str, labels

        return encoded


    def create_inputs(self, masked_text, label):
        item = self.tokenizer(masked_text, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True)
        item = {k: v.to(self.base_model_device) for k, v in item.items()}
        if 'llama' in str(type(self.base_model)).lower() and self.is_instruction_model:
            # LLaMA Instruct Model
            label_token_id = self.tokenizer.tokenizer(f'{label}', add_special_tokens=False)['input_ids']
            if self.tokenizer.tokenizer.decode(label_token_id[0]).strip() == '':
                label_token_id = label_token_id[1:]
        elif self.is_generative:
            label_token_id = self.tokenizer(f' {label}', add_special_tokens=False)['input_ids']
        else:
            label_token_id = self.tokenizer(f'{label}', add_special_tokens=False)['input_ids']
        if not len(label_token_id) == 1:
            # todo we need to support multiple label tokens in the future (e.g., for valence/arousal key words)
            raise ValueError('Only a single label token is supported', label_token_id, label)
        label_token_id = label_token_id[0]

        if self.is_generative:
            item['labels'] = torch.full_like(item['input_ids'], -100)
            last_idx = (item['input_ids'].squeeze() != self.tokenizer.pad_token_id).nonzero()[-1].item()
            item['labels'][:, last_idx] = label_token_id
        else:
            labels = item['input_ids'].clone()
            labels[labels != self.tokenizer.mask_token_id] = -100  # only compute loss on masked tokens
            labels[labels == self.tokenizer.mask_token_id] = label_token_id
            item['labels'] = labels
        return item


    def feature_creator(self, *args, **kwargs):
        if self.use_gradients:
            return self.forward_pass_create_gradients(*args, **kwargs)
        else:
            return self.create_model_embeddings(*args, **kwargs)

    @property
    def feature_creator_id(self):
        return 'grad' if self.use_gradients else 'no-grad'

    def create_model_embeddings(self, inputs):
        inputs = {k: v.to(self.base_model_device) for k, v in inputs.items()}
        inputs = {k: v.unsqueeze(0) if v.ndim == 1 else v for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        mask = input_ids == self.tokenizer.mask_token_id
        labels = inputs['labels']
        mask_labels = labels[labels != -100]
        #input_ids[mask] = mask_labels
        inputs['input_ids'] = input_ids
        del inputs['labels']


        # change mask to [CLS] token, i.e., the first token
        cls_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)
        cls_mask[:, 0] = True


        output = self.base_model.base_model(**inputs)
        #logits = output.logits
        #mask_logits = logits[mask.unsqueeze(-1).expand_as(logits)]
        #return mask_logits

        last_hidden_state = output[0]

        # we currently take only the mean of the last hidden state for the masked tokens
        # note that we apply the mean here to get a similar representation as we have for the gradient input
        return last_hidden_state[cls_mask].mean(dim=0)


# todo rename to create_gradients (once verified nothing breaks when renaming globally)
    def forward_pass_create_gradients(self,
                                      inputs,
                                      return_dict=False,
                                      lr=1e-4,
                                      verbose=False): # todo implement batched=True
        inputs = {k: v.to(self.base_model_device) for k, v in inputs.items()}
        inputs = {k: v.unsqueeze(0) if v.ndim == 1 else (v.squeeze(dim=1) if v.ndim == 3 and v.shape[1] == 1 else v) for k, v in inputs.items()}
        grads = []

        if self.grad_iterations > 1:
            base_model = copy.deepcopy(self.base_model)
            for i in range(self.grad_iterations):
                outputs = base_model(**inputs)
                loss_bert = outputs.loss
                base_model.zero_grad()
                loss_bert.backward()
                gradients = self.gradiend.extract_gradients(base_model, return_dict=True)
                self.base_model.zero_grad(set_to_none=True)
                grads.append(gradients)

                if i < self.grad_iterations - 1:
                    # perform the training step
                    # Step 6: Update the model's weights

                    with torch.no_grad():
                        for name, param in base_model.named_parameters():
                            if param.grad is not None:
                                param.add_(-lr * param.grad)

                # save only the last gradient (for now, maybe add some advanced logic later)
                grads = [grads[-1]]

            if return_dict:
                return grads[-1]
            flatten_gradient = torch.concat(tuple(grad.flatten() for grad in grads[-1].values()))
            return flatten_gradient

        else:


            outputs = self.base_model(**inputs)

            if verbose:
                # Get most likely next token from CausalLMOutputWithPast
                # Find the last valid (non -100) label index for each sequence
                logits = outputs.logits
                input_ids = inputs["input_ids"]


                shape = input_ids.shape
                batch_size = shape[0]
                seq_len = shape[-1]  # always the last dim

                # --- MLM (BERT-style) ---
                if not self.is_generative:
                    mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
                    # batch gather
                    batch_logits = logits[mask_positions[:, 0], mask_positions[:, 1], :]
                    next_token_ids = batch_logits.argmax(dim=-1)
                    next_tokens = self.tokenizer.batch_decode(next_token_ids)

                # --- Causal LM (GPT-style) ---
                else:
                    labels = inputs['labels']
                    pred_mask = torch.roll(labels != -100, shifts=-1, dims=1)
                    batch_logits = logits[pred_mask]
                    next_token_ids = batch_logits.argmax(dim=-1)
                    next_tokens = self.tokenizer.batch_decode(next_token_ids)



                    pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

                    # input_ids shape: [B, T] (or [B, K, T], we just care about last dim)
                    mask = input_ids != pad_id  # [B, T] or [B, K, T]

                    # flatten extra dims if needed
                    mask_2d = mask.view(mask.size(0), -1)  # [B, T']
                    seq_len = mask_2d.size(1)

                    # find last non-pad position for each sequence
                    last_non_pad_indices = mask_2d.int().flip(dims=[1]).argmax(dim=1)
                    last_non_pad_indices = seq_len - 1 - last_non_pad_indices  # shape [B]

                    # now we can index safely
                    batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
                    batch_logits = logits[batch_indices, last_non_pad_indices, :]

                    next_token_ids = batch_logits.argmax(dim=-1)
                    next_tokens = self.tokenizer.batch_decode(next_token_ids)

                print('Predicted next tokens:', next_tokens)

            loss_bert = outputs.loss

            self.base_model.zero_grad()
            loss_bert.backward()
            return self.gradiend.extract_gradients(self.base_model, return_dict=return_dict)

    def forward_pass_create_gradients_(self, inputs, return_dict=False, lr=1e-4, verbose=False):
        # ensure inputs on same device as base model (or move to cpu — see variant B)
        inputs = {k: v.to(self.base_model_device) for k, v in inputs.items()}
        inputs = {k: v.unsqueeze(0) if v.ndim == 1 else (v.squeeze(dim=1) if v.ndim == 3 and v.shape[1] == 1 else v)
                  for k, v in inputs.items()}

        # run forward
        outputs = self.base_model(**inputs)
        loss_bert = outputs.loss

        # collect parameters we care about (only those that require grad)
        params = [p for p in self.base_model.parameters() if p.requires_grad]

        # compute grads via autograd.grad (does NOT create higher-order graph by default)
        grads = torch.autograd.grad(loss_bert, params, create_graph=False, retain_graph=False)

        # detach, clone and optionally move to CPU immediately to free GPU memory
        grads_detached = [g.detach().clone().to('cpu') if g is not None else torch.zeros_like(p, device='cpu')
                          for g, p in zip(grads, params)]

        # optionally zero out base model grads and delete outputs to free memory
        self.base_model.zero_grad(set_to_none=True)
        del outputs
        for t in inputs.values():
            try:
                del t
            except Exception:
                pass
        torch.cuda.empty_cache()
        gc.collect()

        # convert to dict if needed (keyed by parameter name)
        if return_dict:
            named = dict(self.base_model.named_parameters())
            out = {}
            idx = 0
            for name, p in named.items():
                if p.requires_grad:
                    out[name] = grads_detached[idx]
                    idx += 1
            return out
        else:
            # flatten into a single tensor (on CPU)
            return torch.cat([g.flatten() for g in grads_detached])

    @property
    def layers_hash(self):
        return self.gradiend.layers_hash

    # invert the encoded value, i.e. encoded value * (-1), while keeping the decoders value
    def invert_encoding(self, index=None):
        with torch.no_grad():
            if index is None:
                # invert all indices
                self.gradiend.encoder[0].weight.data *= -1
                self.gradiend.encoder[0].bias.data *= -1
                self.gradiend.decoder[0].weight.data *= -1
            else:
                # invert only the index
                if isinstance(index, int):
                    index = [index]
                elif not isinstance(index, list):
                    raise ValueError('Index must be an integer or a list of integers')

                for i in index:
                    if i < 0 or i >= self.gradiend.latent_dim:
                        raise ValueError(f'Index {i} out of bounds for latent dimension {self.gradiend.latent_dim}')
                    self.gradiend.encoder[0].weight.data[i] *= -1
                    self.gradiend.encoder[0].bias.data[i] *= -1
                    self.gradiend.decoder[0].weight.data[i] *= -1


    def save_pretrained(self, save_directory, **kwargs):
        self.gradiend.save_pretrained(save_directory, bert=self.base_model.name_or_path, tokenizer=self.tokenizer.name_or_path, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        load_directory,
                        layers=None,
                        latent_dim=1,
                        torch_dtype=torch.float32,
                        device=None,
                        use_gradients=True,
                        init_gradiends=None,
                        **kwargs):
        assert isinstance(load_directory, str) or hasattr(load_directory, 'name_or_path'), 'load_directory must be a string or an AutoModelForLM instance, but got: {}'.format(type(load_directory))

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_encoder = device
        device_decoder = device
        device_base_model = device

        if init_gradiends:
            if latent_dim is not None and latent_dim != len(init_gradiends):
                raise ValueError('If init_gradiends is provided, latent_dim must be None or equal to init_gradiends')

            base_model_id = load_directory.split('/')[-1]
            gradiends = [GradiendModel.from_pretrained(f'{d}/{base_model_id}') for d in init_gradiends]
            for g in gradiends:
                if g.latent_dim != 1:
                    raise ValueError('Only Gradiend models with latent_dim=1 can be used for initialization')

            base_models = set(g.kwargs['base_model'] for g in gradiends)
            if len(base_models) != 1:
                raise ValueError('All Gradiend models used for initialization must have the same base_model, got:', base_models)


            if load_directory not in base_models:
                raise ValueError(f'The provided base model id {load_directory} does not match the base model ids of the Gradiend models used for initialization: {base_models}')

            base_model = AutoModelForLM.from_pretrained(load_directory)
            tokenizer = AutoTokenizerForLM.from_pretrained(load_directory)

            gradiend = CombinedGradiendModel(gradiends)
            layers = gradiend.layers
        else:
            layers = layers or []
            if len(layers) == 1 and isinstance(layers[0], list):
                layers = layers[0]

            load_directory_str = load_directory if isinstance(load_directory, str) else load_directory.name_or_path
            if 'llama' in load_directory_str.lower():
                # check that two GPUs are available
                cuda_count = torch.cuda.device_count()
                if cuda_count < 2:
                    raise RuntimeError("Two GPUs are required for GRADIEND Llama models.")

                device_encoder = torch.device("cuda:0")
                device_decoder = torch.device("cuda:1")

                available_gpu_ram = torch.cuda.get_device_properties(device_encoder).total_memory // (1024 ** 3)
                available_gpu_ram_greater_than_100GB = available_gpu_ram >= 100
                device_base_model = device_encoder if (cuda_count == 2 or available_gpu_ram_greater_than_100GB) else torch.device("cuda:2")

            try:
                gradiend = GradiendModel.from_pretrained(load_directory, device_encoder=device_encoder, device_decoder=device_decoder)

                if layers and gradiend.layers != layers:
                    raise ValueError(f'The provided layers {layers} do not match the layers in the model configuration {gradiend.layers}')
                else:
                    layers = gradiend.layers

                if 'config' in gradiend.kwargs['training']:
                    base_model_id = gradiend.kwargs['training']['config']['base_model']
                else:
                    # todo deprecate this format
                    base_model_id = gradiend.kwargs['training']['base_model']
                base_model = AutoModelForLM.from_pretrained(base_model_id, torch_dtype=torch_dtype).to(device_base_model)
                tokenizer = gradiend.kwargs.get('tokenizer', base_model_id)
                tokenizer = AutoTokenizerForLM.from_pretrained(tokenizer)
            except (FileNotFoundError, TypeError, ValueError):
                print('No model with GRADIEND found in the specified directory:', load_directory, ' -> creating a new GRADIEND')

                if isinstance(load_directory, str):
                    base_model = AutoModelForLM.from_pretrained(load_directory, torch_dtype=torch_dtype).to(device_base_model)
                    tokenizer = AutoTokenizerForLM.from_pretrained(load_directory)
                else:
                    base_model = load_directory
                    tokenizer = AutoTokenizerForLM.from_pretrained(base_model.name_or_path)

                excluded_layers = {
                    'cls.prediction', # todo check if this is still needed as we use base_model (e.g. ,BertModel, rather than BertForMaskedLM)
                    'lm_head',
                    'pooler.dense',
                }

                layer_map = {k: v for k, v in base_model.named_parameters() if not any(x in k.lower() for x in excluded_layers)}

                if layers:
                    if not isinstance(layers, dict):
                        # layers information are provided
                        # check if layer description can be matched with layer_map keys
                        # layer description could also be "*.layer.10*"
                        matched_layers = []
                        for layer in layers:
                            if layer in layer_map:
                                matched_layers.append(layer)
                            else:
                                # Handle wildcard matching (e.g., "*.layer.10*")
                                layer_pattern = layer.replace('.', r'\.').replace('*', r'.*')  # Convert to regex pattern
                                layer_regex = re.compile(layer_pattern)

                                for layer_name in layer_map:
                                    if layer_regex.fullmatch(layer_name):
                                        matched_layers.append(layer_name)

                        layers = list(sorted(matched_layers, key=lambda x: list(layer_map.keys()).index(x)))
                else:
                    # no layer provided, i.e. all layers are used that are part of the core model, i.e. all layers that are not part of prediction layers
                    layers = [layer for layer in layer_map]

                if use_gradients:
                    if isinstance(layers, dict):
                        input_dim = sum([v.sum() for v in layers.values()])
                    else:
                        input_dim = sum([layer_map[layer].numel() for layer in layers])
                else:
                    # use the hidden size of the base model as input dimension
                    input_dim = base_model.config.hidden_size

                gradiend = GradiendModel(input_dim,
                                   layers=layers,
                                   latent_dim=latent_dim,
                                   base_model=load_directory,
                                   torch_dtype=torch_dtype,
                                   device_encoder=device_encoder,
                                   device_decoder=device_decoder,
                                   **kwargs)

        # freeze all layers that do not require gradient calculations
        freeze_layers_until_target(base_model, *layers)

        model = ModelWithGradiend(base_model, gradiend, tokenizer, base_model_device=device_base_model, use_gradients=use_gradients)
        model.name_or_path = load_directory
        return model

    def ae_named_parameters(self, part='all'):
        idx = 0
        if part == 'all':
            yield from self.gradiend.named_parameters()
            return
        elif part == 'encoder':
            layer_map = {k: v for k, v in self.gradiend.encoder.named_parameters()}
            weights = layer_map['0.weight'].squeeze()
        elif 'decoder' in part:
            layer_map = {k: v for k, v in self.gradiend.decoder.named_parameters()}
            if part == 'decoder':
                weights = layer_map['0.weight'].squeeze()
            elif part == 'decoder-sum':
                weights = (layer_map['0.weight'] + layer_map['0.bias']).squeeze()
            elif part == 'decoder-bias':
                weights = layer_map['0.bias'].squeeze()
            else:
                raise ValueError('Unsupported part:', part)
        else:
            raise ValueError('Unsupported part:', part)

        for layer in self.gradiend.layers:
            orig_shape = self.layer_map[layer].shape
            num_elements = orig_shape.numel()
            yield layer, weights[idx:idx + num_elements].reshape(orig_shape)
            idx += num_elements
        if idx != weights.numel():
            raise ValueError(f'Inconsistent number of elements in the weights and expected number of elements in the layers ({idx} vs. {weights.numel()})')


if __name__ == '__main__':
    gradiend = ModelWithGradiend.from_pretrained('results/models/bert-base-cased')