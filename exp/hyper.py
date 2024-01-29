from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class SBN_Experiment:
    """Dataclass for storing all the hyperparameters of an experiment."""

    early_stopping: bool
    # Number of epoch to train for
    epoch: int
    batch_size: int

    # Type of training data
    train_data: str

    # optimizer arguments like lr ect.
    optimizer_kwargs: Dict[str, Any]
    # optimizer class
    optimizer_cls: object

    # Language used for the train,val and test data
    lang: str

    # Number of train epochs before validating
    # So if set to 2 the model will be evaluated after train epoch: 2,4,6...
    val_epoch: int

    # Model name from hugging fac
    model_name: str
    # Tokenizer name from hugging face
    tokenizer_name: str

    # Generated Name run
    name_run: str = field(init=False, default_factory=lambda: "")

    dict = asdict

    def __post_init__(self):
        self.name_run = "_".join(
            [
                self.model_name,
                self.lang,
                self.train_data,
                "_".join(f"{k}_{v}" for k, v in self.optimizer_kwargs.items()),
            ]
        ).replace("/", "_")

    def select_kwargs(self, datacls):
        return {
            k: getattr(self, k)
            for k in set(self.__dataclass_fields__.keys()).intersection(
                set(datacls.__dataclass_fields__.keys())
            )
        }


def dict_to_string(input_dict):
    return "{" + ", ".join(f"{k}: {v!r}" for k, v in input_dict.items()) + "}"
