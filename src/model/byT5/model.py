
import os
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, T5ForConditionalGeneration, AdamW

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


Config = {"batch_size": 8,
          "cuda_index": 0,
          "max_length": 512
          }


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_file_path):
        # combine input: original text with masked sbn
        print("Reading lines...")
        with open(input_file_path, encoding="utf-8") as f:
            self.text = f.readlines()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx].split("\t")[0]
        sbn = self.text[idx].split("\t")[1].replace("\n", "")
        return text, sbn


def get_dataloader(input_file_path, batch_size=Config["batch_size"]):
    dataset = Dataset(input_file_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


class Generator:

    def __init__(self, lang, load_path=""):
        """
        :param train: train or test
        """
        self.device = torch.device(f"cuda:{Config['cuda_index']}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-base', max_length=Config["max_length"])

        if len(load_path) == 0:
            self.model = T5ForConditionalGeneration.from_pretrained('google/byt5-base', max_length=Config["max_length"])
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(load_path)

        self.model.to(self.device)

    def evaluate(self, val_loader, save_path):
        with open(save_path, 'w', encoding="utf-8") as f:
            self.model.eval()
            with torch.no_grad():
                for i, (text, target) in enumerate(tqdm(val_loader)):
                    x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=Config["max_length"])['input_ids'].to(
                        self.device)
                    out_put = self.model.generate(x)
                    for j in range(len(out_put)):
                        o = out_put[j]
                        pred_text = self.tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        f.write(pred_text)
                        f.write('\n')

    def train(self, train_loader, val_loader, lr, epoch_number):
        optimizer = AdamW(self.model.parameters(), lr)
        for epoch in range(epoch_number):
            self.model.train()
            pbar = tqdm(train_loader)
            for batch, (text, target) in enumerate(pbar):
                x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=Config["max_length"])['input_ids'].to(
                    self.device)
                y = self.tokenizer(target, return_tensors='pt', padding=True, truncation=True, max_length=Config["max_length"])['input_ids'].to(
                    self.device)

                optimizer.zero_grad()
                output = self.model(x, labels=y)
                loss = output.loss
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Loss: {format(loss.item(), '.3f')}")
