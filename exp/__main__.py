from exp import wrapper, dataset, hyper, parser
import torch
from pathlib import Path
import wandb

torch.cuda.empty_cache()


def run_exp(name: str, wandb_bool: bool, print_bool: bool):
    print(f"running : {name}")
    conf = hyper.SBN_Experiment(
        model_name=name,
        tokenizer_name=name,
        epoch=10,
        val_epoch=4,
        tokenizer_kwargs=dict(
            max_length=256,
        ),
        tokenizer_call_args=dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ),
        decoder_call_args=dict(
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        model_kwargs=dict(
            max_length=256,
        ),
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=dict(
            lr=0.0001,
        ),
        train_data=args.train,
        lang=lang,
    )

    logger = wrapper.Print_Logger if print_bool else wrapper.No_Logger
    if wandb_bool:
        wandb.init(
            entity="comp_sem",
            project="first_run",
            config=conf,
        )
        logger = wandb

    wmodel = wrapper.Wrapper(
        **conf.select_kwargs(wrapper.Wrapper),
        logger=logger,
    )

    print("Training")
    wmodel.train(train_dataloader, dev_dataloader)
    print("Eval")
    rsave_path = Path.cwd() / "results" / name / lang
    rsave_path.mkdir(parents=True, exist_ok=True)
    wmodel.evaluate(test_dataloader, rsave_path / "test.txt")

    msave_path = Path.cwd() / "models" / name / lang
    msave_path.mkdir(parents=True, exist_ok=True)

    wmodel.model.save_pretrained(msave_path)

    if wandb_bool:
        wandb.finish()


args = parser.create_arg_parser()
print(args.wandb)

# train process
lang = args.lang
batch_size = 10
data_path = lambda x, y, lang=lang: dataset.get_data_path(lang, x, y)

train_dataloader = dataset.get_dataloader(
    data_path("train", args.train), batch_size=batch_size
)
test_dataloader = dataset.get_dataloader(
    data_path("test", args.test), batch_size=batch_size
)
dev_dataloader = dataset.get_dataloader(
    data_path("dev", args.dev), batch_size=batch_size
)

names = [
    "GermanT5/t5-efficient-gc4-all-german-small-el32",
    "sonoisa/t5-base-japanese",
    "yhavinga/t5-base-dutch",
    "google/flan-t5-base",
    "gsarti/it5-base",  # en batch_size 5 outof ram
    # "google/mt5-base",
    # "GermanT5/t5-efficient-gc4-all-german-large-nl36",
]

run_exp(names[0], wandb_bool=args.wandb, print_bool=args.print)

# for name in names:
#     run_exp(name)
