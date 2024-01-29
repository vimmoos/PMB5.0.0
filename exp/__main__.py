from exp import wrapper, dataset, hyper, parser, metric, smatch_func
import torch
from pathlib import Path
import wandb

torch.cuda.empty_cache()


args = parser.create_arg_parser()

# train process
lang = args.lang
batch_size = args.batch_size
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

print(f"running : {args.model_name}")
conf = hyper.SBN_Experiment(
    early_stopping=args.early_stop,
    model_name=args.model_name,
    tokenizer_name=args.tokenizer_name or args.model_name,
    epoch=args.epoch,
    val_epoch=args.val_epoch,
    optimizer_cls=getattr(torch.optim, args.optimizer),
    optimizer_kwargs=dict(
        lr=args.learning_rate,
    ),
    train_data=args.train,
    lang=lang,
    batch_size=args.batch_size,
)

logger = wrapper.Print_Logger if args.print else wrapper.No_Logger

experiment_name = conf.name_run
if args.wandb:
    wandb.init(
        entity="comp_sem",
        project=args.wandb_project,
        config=conf,
    )
    logger = wandb
    experiment_name = f"{conf.name_run}_{wandb.run.path.replace('/','_')}"

wmodel = wrapper.Wrapper(
    **conf.select_kwargs(wrapper.Wrapper),
    val_metrics=[
        smatch_func.compute_smatchpp,
        metric.hamming_dist,
        metric.similarity,
    ],
    logger=logger,
)
print(args)
print(conf)
print(wmodel)

print("Training")
wmodel.train(train_dataloader, dev_dataloader)
print("Eval")

rsave_path = Path.cwd() / "results"
rsave_path.mkdir(parents=True, exist_ok=True)
wmodel.evaluate(
    test_dataloader,
    rsave_path / experiment_name + ".txt",
)

msave_path = Path.cwd() / "models" / experiment_name
msave_path.mkdir(parents=True, exist_ok=True)

wmodel.model.save_pretrained(msave_path)

if args.wandb:
    wandb.finish()


# names = [
#     "GermanT5/t5-efficient-gc4-all-german-small-el32",
#     "sonoisa/t5-base-japanese",
#     "yhavinga/t5-base-dutch",
#     "google/flan-t5-base",
#     "gsarti/it5-base",  # en batch_size 5 outof ram
#     # "google/mt5-base",
#     # "GermanT5/t5-efficient-gc4-all-german-large-nl36",
# ]
