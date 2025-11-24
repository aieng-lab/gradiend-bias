import argparse

from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers.modeling_outputs import MaskedLMOutput

from gradiend.model import ModelForClassificationWithMLM
from gradiend.setups.gender.en import GenderEnSetup
from gradiend.training.gradiend_training import train_for_configs

from transformers import BertForSequenceClassification, BertForMaskedLM, BertConfig
import torch.nn as nn

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertForMaskedLM

def run_analysis(model_name, mlm_model):
    setup = GenderEnSetup()

    #mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
    #finetuned_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    #finetuned_state = finetuned_model.state_dict()
    #mlm_model.load_state_dict(finetuned_state, strict=False)
    print("Loading model:", model_name)
    model = ModelForClassificationWithMLM.from_pretrained(cls_checkpoint=model_name, mlm_checkpoint=mlm_model)

    config = {model: dict()}
    train_for_configs(setup, config, n=1, version='finetuning-analysis')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning Analysis')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model')
    parser.add_argument('--mlm_model', type=str, required=True, help='Name of the MLM model for the MLM head')
    args = parser.parse_args()
    run_analysis(args.model_name, args.mlm_model)