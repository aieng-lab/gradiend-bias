import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig

from gradiend.setups.gender.en import read_genter, read_namexact
from gradiend.training.decoder_only_mlm.model import DecoderModelWithMLMHead

class SimpleMLMDataset(Dataset):
    def __init__(self, tokenizer, texts, targets):
        self.tokenizer = tokenizer
        self.texts = texts
        self.targets = targets

        assert len(self.texts) == len(self.targets), "Texts and targets must have the same length."

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=48)
        input_ids = enc.input_ids.squeeze(0)
        attn_mask = enc.attention_mask.squeeze(0)
        labels = self.targets[idx]
        labels = self.tokenizer(labels, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
        return input_ids, attn_mask, labels


class MLMDataset(Dataset):
    def __init__(self, tokenizer, texts, names):
        self.tokenizer = tokenizer
        self.texts = texts
        self.names = names

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        random_name = self.names.iloc[idx % len(self.names)]
        name = random_name['name']
        gender = random_name['gender']
        text = text.replace("[PRONOUN]", '[MASK]').replace("[NAME]", name)
        labels = 'he' if gender == 'M' else 'she'
        enc = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=48)
        input_ids = enc.input_ids.squeeze(0)
        attn_mask = enc.attention_mask.squeeze(0)
        labels = self.tokenizer(labels, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
        return input_ids, attn_mask, labels

def train(
        base_model: str = "gpt2",

        # training parameters
        batch_size: int = 16,
        epochs: int = 10,
        lr: float = 5e-4,
):

    # ======= Config =======
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = base_model.split("/")[-1]
    output_path = f'results/decoder-mlm-head/{model_id}'


    # ======= Prepare Tokenizer =======
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    tokenizer.pad_token = tokenizer.eos_token

    target_token_ids = tokenizer.convert_tokens_to_ids(["he", "she"])
    print("Target token IDs:", target_token_ids)

    # ======= Example Training Data =======
    genter = read_genter()
    genter = genter[genter['pronoun_count'] == 1]
    # todo use equal amount of he and she
    # todo use only train names

    train_texts = genter['masked'].tolist()
    #train_texts = [t.replace("[PRONOUN]", '[MASK]').replace("[NAME]", name) for t, name in zip(train_texts, genter['name'])]
    #targets = genter['pronoun'].tolist()

    names = read_namexact(split='train')
    train_dataset = MLMDataset(tokenizer, train_texts, names)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ======= Model =======

    model = DecoderModelWithMLMHead.from_pretrained(
        base_model,
        mask_token_id=tokenizer.mask_token_id,
        target_token_ids=target_token_ids
    )
    model.decoder.resize_token_embeddings(len(tokenizer))  # In case we added [MASK]

    # Freeze decoder
    for p in model.decoder.parameters():
        p.requires_grad = False

    model.to(DEVICE)

    # ======= Optimizer =======
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)

    # ======= Training Loop =======
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, attn_mask, labels = [b.to(DEVICE) for b in batch]

            # labels: we only care about the mask position
            # In restricted mode, label must be the target token ID
            output = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)

            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # ======= Save Model =======
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    train()