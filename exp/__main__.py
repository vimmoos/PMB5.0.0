from exp import wrapper, dataset, hyper, parser, metric, smatch_func
import torch
from pathlib import Path
import wandb
import gc

torch.cuda.empty_cache()


def gen_data(batch_size, lang):
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
    return train_dataloader, test_dataloader, dev_dataloader


def single_run(args):
    # train process
    train_dataloader, test_dataloader, dev_dataloader = gen_data(
        args.batch_size,
        args.lang,
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
        lang=args.lang,
        batch_size=args.batch_size,
    )

    logger = wrapper.Print_Logger if args.print else wrapper.No_Logger

    experiment_name = conf.name_run
    if args.wandb:
        run = wandb.init(
            entity="comp_sem",
            project=args.wandb_project,
            config=conf,
        )
        logger = run
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

    print("Training")
    batch_size = args.batch_size
    while True:
        try:
            wmodel.train(train_dataloader, dev_dataloader)
        except torch.cuda.OutOfMemoryError:
            batch_size //= 2

            if batch_size <= 1:
                print("Cannot run this model no matter the batch_size")
                if args.wandb:
                    run.finish()
                    exit()
            if args.wandb:
                run.finish()
                run = wandb.init(
                    entity="comp_sem",
                    project=args.wandb_project,
                    config=conf,
                )
                logger = run
                run.config["final_batch_size"] = batch_size

            torch.cuda.empty_cache()
            gc.collect()
            train_dataloader, dev_dataloader, _ = gen_data(batch_size, args.lang)
            wmodel = wrapper.Wrapper(
                **conf.select_kwargs(wrapper.Wrapper),
                val_metrics=[
                    smatch_func.compute_smatchpp,
                    metric.hamming_dist,
                    metric.similarity,
                ],
                logger=logger,
            )
            print(f"OutOfMemoryError Trying with batch_size {batch_size}")
            continue
        break
    print("Eval")

    rsave_path = Path.cwd() / "results"
    rsave_path.mkdir(parents=True, exist_ok=True)
    wmodel.evaluate(
        test_dataloader,
        rsave_path / f"{experiment_name}.txt",
    )

    msave_path = Path.cwd() / "models" / experiment_name
    msave_path.mkdir(parents=True, exist_ok=True)

    wmodel.model.save_pretrained(msave_path)

    if args.wandb:
        run.finish()


args = parser.create_arg_parser()
allowed_model = hyper.multilingual + hyper.lang_to_model[args.lang]
if args.model_name in allowed_model:
    single_run(args)
else:
    if args.wandb:
        wandb.init()
        wandb.finish()
    print("Invalid parameters configuration")
