from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class SBN_Experiment:
    # early_stopping: bool
    epoch: int

    train_data: str

    optimizer_kwargs: Dict[str, Any]
    optimizer_cls: object

    tokenizer_kwargs: Dict[str, Any]
    tokenizer_call_args: Dict[str, Any]
    decoder_call_args: Dict[str, Any]

    model_kwargs: Dict[str, Any]

    lang: str

    val_epoch: int

    model_name: str
    tokenizer_name: str

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
