from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class SBNDataset(Dataset):
    """Dataset class for SBN type of data."""

    def __init__(self, input_file_path: Path):
        # combine input: original text with masked sbn
        print(f"Reading lines from {input_file_path}")
        with open(input_file_path, encoding="utf-8") as f:
            self.text = f.readlines()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx].split("\t")[0]
        sbn = self.text[idx].split("\t")[1].replace("\n", "")
        return text, sbn


def get_dataloader(
    input_file_path: Path,
    batch_size: int = 10,
    **data_loader_kwargs,
):
    return DataLoader(
        SBNDataset(input_file_path),
        batch_size=batch_size,
        **data_loader_kwargs,
    )


def get_data_path(lang: str, split: str, version: str):
    """Return the path of some data in the "data" folder of this project."""
    base_path = Path.cwd()
    return base_path / "data" / "pmb" / "seq2seq" / lang / split / f"{version}.sbn"
