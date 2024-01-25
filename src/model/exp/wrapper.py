from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Union, Dict, Any, Optional, Callable, List
from pathlib import Path
import torch as th
from tqdm import tqdm, trange
from src.model.exp.metric import hamming_dist, similarity
import numpy as np


class No_Logger:
    def log(*args, **kwargs):
        pass


class Print_Logger:
    def log(*args, **_):
        print(*args)


@dataclass
class Wrapper:
    model_name: Union[str, Path]
    tokenizer_name: Optional[Union[str, Path]] = None
    epoch: int = 10

    val_epoch: int = 2
    val_metrics: List[Callable[[str, str], float]] = field(
        default_factory=lambda: [hamming_dist, similarity]
    )

    device: str = "cuda"
    logger: Optional[Any] = None

    tokenizer_cls: object = field(default_factory=lambda: AutoTokenizer)
    tokenizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            max_length=256,
        )
    )
    tokenizer_call_args: Dict[str, Any] = field(
        default_factory=lambda: dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
    )
    decoder_call_args: Dict[str, Any] = field(
        default_factory=lambda: dict(
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )

    model_cls: object = field(default_factory=lambda: AutoModelForSeq2SeqLM)
    model_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            max_length=256,
        )
    )

    optimizer_cls: object = field(default_factory=lambda: th.optim.AdamW)
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            lr=0.0001,
        )
    )

    def __post_init__(self):
        self.logger = self.logger or No_Logger()
        self.tokenizer_name = self.tokenizer_name or self.model_name
        self.tokenizer = self.tokenizer_cls.from_pretrained(
            self.tokenizer_name, **self.tokenizer_kwargs
        )

        self.model = self.model_cls.from_pretrained(
            self.model_name, **self.model_kwargs
        )
        self.model.to(self.device)
        self.optimizer = self.optimizer_cls(
            self.model.parameters(), **self.optimizer_kwargs
        )

    def tok(self, text):
        return self.tokenizer(
            text,
            **self.tokenizer_call_args,
        )["input_ids"].to(
            self.device,
        )

    def decode(self, text):
        return self.tokenizer.decode(text, **self.decoder_call_args)

    def eval_metrics(self, pred, target):
        metrics = np.array(
            [
                [m(pred, target) for m in self.val_metrics]
                for pred, target in zip(pred, target)
            ]
        )
        return {
            f"val/{m.__name__}": metrics[:, i].mean()
            for i, m in enumerate(self.val_metrics)
        }

    def evaluate(
        self,
        test_loader,
        save_path: Optional[Union[str, Path]] = None,
    ):
        self.model.eval()
        pred_text = []
        target_text = []
        with th.no_grad():
            for text, target in tqdm(test_loader):
                x = self.tok(text)
                out_put = self.model.generate(x)
                pred_text.extend([self.decode(out) for out in out_put])
                target_text.extend(list(target))

        if save_path is None:
            out_put = self.eval_metrics(pred_text, target_text)
            self.logger.log(out_put)
            return out_put

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(pred_text))

    def train(self, train_loader, val_loader):
        for epoch in trange(self.epoch):
            self.model.train()
            for text, target in train_loader:
                x = self.tok(text)
                y = self.tok(target)
                output = self.model(x, labels=y)
                loss = output.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.logger.log({"train/loss": loss.detach().item()})

            if epoch > 2 and epoch % self.val_epoch == 0:
                self.evaluate(val_loader)
