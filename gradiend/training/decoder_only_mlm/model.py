import json
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import Parameter
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM, AutoModel,
)
from transformers.utils import ModelOutput
from typing import Optional, Union, List, Iterator, Tuple
import os


@dataclass
class DecoderWithMLMHeadOutput(ModelOutput):
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class DecoderModelWithMLMHead(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig, target_token_ids: Optional[List[int]] = None):
        super().__init__(config)
        # Base decoder model
        self.decoder = AutoModelForCausalLM.from_config(config)

        self.config.model_type = f'{self.decoder.config.model_type}-with-mlm-head'
        #AutoConfig.register(self.config.model_type, self.config_class)
        #AutoModel.register(self.config.model_type, self.__class__)

        self.target_token_ids = target_token_ids

        if self.target_token_ids is None:
            self.classifier = self.decoder.lm_head
        else:
            hidden_size = self.decoder.config.hidden_size
            self.classifier = nn.Linear(hidden_size, len(self.target_token_ids))
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)

        self.init_weights()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        target_token_ids: Optional[List[int]] = None,
        mask_token_id: int = None,
        *model_args,
        **kwargs
    ):
        # Detect if this is a custom checkpoint (by presence of our special meta file)
        meta_path = os.path.join(pretrained_model_name_or_path, "config_mlm_head.json")
        is_custom_checkpoint = os.path.exists(meta_path)

        if is_custom_checkpoint:
            # --- Load from custom checkpoint ---
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            config.mask_token_id = mask_token_id or getattr(config, "mask_token_id", None)

            # Restore target_token_ids from meta file if not provided
            with open(meta_path) as f:
                meta = json.load(f)

            if target_token_ids is None and "target_token_ids" in meta:
                target_token_ids = meta["target_token_ids"]

            # Init wrapper
            model = cls(config, target_token_ids=target_token_ids)

            # Load saved weights
            weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu")
            wte_size = state_dict['decoder.transformer.wte.weight'].size(0)
            # resize embeddings if needed
            current_vocab_size = model.decoder.config.vocab_size
            if current_vocab_size != wte_size:
                model.decoder.resize_token_embeddings(wte_size)
            model.load_state_dict(state_dict, strict=True)

        else:
            # --- Load from a standard LM checkpoint (e.g., "gpt2") ---
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            config.mask_token_id = mask_token_id or getattr(config, "mask_token_id", None)

            # Init wrapper — this will internally freeze decoder and init head
            model = cls(config, target_token_ids=target_token_ids)

            # Load decoder from the base LM checkpoint
            model.decoder = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

            # Replace classifier if in full vocab mode
            if target_token_ids is None:
                model.classifier = model.decoder.lm_head

        return model


    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        # Save wrapper config
        self.config.save_pretrained(save_directory)

        # Save wrapper-specific meta
        meta = {"target_token_ids": self.target_token_ids}
        with open(os.path.join(save_directory, "config_mlm_head.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Save all model weights (wrapper + decoder)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    def forward(
            self,
            input_ids: Union[torch.LongTensor, List[int], List[List[int]]],
            attention_mask: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None
    ):
        # Ensure tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, device=self.device)

        # Ensure 2D (batch, seq_len)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if labels is not None and labels.dim() == 1:
                labels = labels.unsqueeze(0)

        # Check for exactly one [MASK] per sequence
        mask_token_id = getattr(self.config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("mask_token_id must be set in the config for MLM forward pass.")

        #mask_counts = (input_ids == mask_token_id).sum(dim=1)
        #if not mask_counts.eq(1).all():
        #    warnings.warn(
        #        f"Some sequences do not contain exactly one [MASK] token "
        #        f"(counts: {mask_counts.tolist()}). Proceeding anyway."
        #    )

        outputs = self.decoder.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        mask_pos = (input_ids == mask_token_id).nonzero(as_tuple=False)

        logits_list = []
        for batch_idx, seq_idx in mask_pos:
            h = hidden_states[batch_idx, seq_idx, :]
            logits = self.classifier(h)
            logits_list.append(logits)

        if logits_list:
            logits = torch.stack(logits_list, dim=0)
        else:
            logits = torch.empty(
                (0, len(self.target_token_ids) if self.target_token_ids else self.config.vocab_size),
                device=hidden_states.device
            )

        loss = None
        if labels is not None and logits.numel() > 0:
            loss_fct = nn.CrossEntropyLoss()

            if self.target_token_ids is None:
                selected_labels = labels[mask_pos[:, 0]]
                loss = loss_fct(logits, selected_labels)
            else:
                label_map = {tid: idx for idx, tid in enumerate(self.target_token_ids)}
                if labels.shape[1] == 1:
                    selected_labels = torch.tensor([l  for i, l in enumerate(labels) for _ in range((mask_pos[:, 0] == i).sum())])
                else:
                    selected_labels = labels[mask_pos[:, 0], mask_pos[:, 1]]
                    #print(selected_labels)
                    #mapped_labels = torch.tensor(
                    #    [label_map.get(l.item(), -1) for l in selected_labels],
                    #    device=logits.device
                    #)
                    #valid_mask = mapped_labels != -1
                    #if valid_mask.any():
                    #    loss = loss_fct(logits[valid_mask], mapped_labels[valid_mask])

                mapped_labels = torch.tensor(
                    [label_map.get(l.item(), -1) for l in selected_labels],
                    device=logits.device
                )

                valid_mask = mapped_labels != -1
                if valid_mask.any():
                    logits = logits[valid_mask]
                    mapped_labels = mapped_labels[valid_mask]
                    loss = loss_fct(logits, mapped_labels)
                else:
                    # create dummy loss
                    loss = torch.tensor(0.0, requires_grad=True)

        return DecoderWithMLMHeadOutput(logits=logits, loss=loss)


    def named_parameters(
            self,
            prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        """
        Override to ensure we return only the decoder parameters and not the head.
        """
        return self.decoder.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate
        )