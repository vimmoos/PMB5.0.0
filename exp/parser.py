import argparse

MODEL_OPTIONS = [
    "t5-base",              # model used for English and German 
    "gsarti/it5-base",      # model used for Italian (lang = it)
    "yhavinga/t5-base-dutch", # model used fro Dutch (lang = nl)
    "google/mt5-base",      # multilingual model no1
    "google/flan-t5-base",  # multilingual model no2
]

def parse_args():
    """Parse command line aguments of main()"""

    parser = argparse.ArgumentParser()

    # --- args for language and data splits ---
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        choices=['en', 'nl', 'de', 'it'],
        required=True,
        help="language in [en, nl, de ,it]",
    )

    parser.add_argument(
        "-tr", 
        "--train-split", 
        type=str,
        choices=['gold', 'silver', 'bronze', 'copper'],
        # nargs='+',
        required=True,
        help="data split to fine-tune the model on",
    )

    parser.add_argument(
        "-d", 
        "--dev-split",
        type=str,
        default="standard",
        required=False,
        help="data split to use as dev set"
    )

    parser.add_argument(
        "-te",
        "--test-split",
        type=str,
        default="standard",
        required=False,
        help="data split to use as test set"
    )

    # --- args for model and its parameters ---
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        required=False,
        help="Tokenizer name, if not provided the model tokenizer will be used if available",
    )

    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        choices=MODEL_OPTIONS,
        required=True,
        help="Model name, must belong to some model in the transformers library",
    )

    parser.add_argument(
        "--early_stop",
        action='store_true',
        default=True,
        required=False,
        help="use early stop mechanism",
    )
    
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        required=False,
        help="number of training epochs",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=10,
        required=False,
        help="batch size for training and testing",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0001,
        required=False,
        help="learing rate for the optimizer",
    )

    parser.add_argument(
        "-opt",
        "--optimizer",
        type=str,
        default="AdamW",
        required=False,
        help="optimizer class, must belong to torch.optim",
    )

    parser.add_argument(
        "--val_epoch",
        type=int,
        default=4,
        required=False,
        help="Number of train epochs before validating",
    )

    # --- args for logging ---
    parser.add_argument(
        "--wandb",
        action='store_true',
        default=False,
        required=False,
        help="use wandb as a logger",
    )

    parser.add_argument(
        "--wandb_project",
        default="default_name",
        type=str,
        required=False,
        help="the project to be used with wandb",
    )

    parser.add_argument(
        "--print",
        action='store_true',
        required=False,
        help="use base print logger",
    )

    return parser.parse_args()