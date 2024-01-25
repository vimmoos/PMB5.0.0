from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class SBN_Experiment:
    """Dataclass for storing all the hyperparameters of an experiment."""

    # early_stopping: bool
    # Number of epoch to train for
    epoch: int

    # Type of training data
    train_data: str

    # optimizer arguments like lr ect.
    optimizer_kwargs: Dict[str, Any]
    # optimizer class
    optimizer_cls: object

    # tokenizer arguments
    # Most of the time they are the default
    tokenizer_kwargs: Dict[str, Any]
    tokenizer_call_args: Dict[str, Any]
    decoder_call_args: Dict[str, Any]

    model_kwargs: Dict[str, Any]

    # Language used for the train,val and test data
    lang: str

    # Number of train epochs before validating
    # So if set to 2 the model will be evaluated after train epoch: 2,4,6...
    val_epoch: int

    # Model name from hugging face
    model_name: str
    # Tokenizer name from hugging face
    tokenizer_name: str

    # Generated Name run
    name_run: str = field(init=False)

    dict = asdict

    def __post_init__(self):
        pass

    def select_kwargs(self, datacls):
        return {
            k: getattr(self, k)
            for k in set(self.__dataclass_fields__.keys()).intersection(
                set(datacls.__dataclass_fields__.keys())
            )
        }
