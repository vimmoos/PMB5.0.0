import wrapper, dataset, hyper, parser, metric, smatch_func
import torch
from pathlib import Path
import wandb
import os

torch.cuda.empty_cache()


args = parser.parse_args()

# train process
lang = args.language
batch_size = args.batch_size
data_path = lambda x, y, lang=lang: dataset.get_data_path(lang, x, y)

train_dataloader = dataset.get_dataloader(
    data_path("train", args.train_split), batch_size=batch_size
)
test_dataloader = dataset.get_dataloader(
    data_path("test", args.test_split), batch_size=batch_size
)
dev_dataloader = dataset.get_dataloader(
    data_path("dev", args.dev_split), batch_size=batch_size
)

print(f"running : {args.model_name}")
conf = hyper.SBN_Experiment(
    early_stopping=args.early_stop,
    model_name=args.model_name,
    tokenizer_name=args.tokenizer_name or args.model_name,
    epoch=args.epochs,
    val_epoch=args.val_epoch,
    optimizer_cls=getattr(torch.optim, args.optimizer),
    optimizer_kwargs=dict(
        lr=args.learning_rate,
    ),
    train_data=args.train_split,
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

print(f"---- Fine-tuning {args.model_name} ----")
wmodel.train(train_dataloader, dev_dataloader)

print("---- Evaluating model ----")
rsave_path = Path.cwd() / "results"
rsave_path.mkdir(parents=True, exist_ok=True)
wmodel.evaluate(
    test_dataloader,
    rsave_path / f"{experiment_name}.txt",
)

try:
    msave_path = Path(os.environ['MODEL_SAVEPATH'] + experiment_name)
except KeyError:
    msave_path = Path.cwd() / "models" / experiment_name
    
msave_path.mkdir(parents=True, exist_ok=True)

wmodel.model.save_pretrained(msave_path)

if args.wandb:
    wandb.finish()
