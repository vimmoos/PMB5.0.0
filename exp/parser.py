import argparse

# TODO
# Create the arguments parserhttps://github.com/vimmoos/PMB5.0.0/tree/main/exp


def create_arg_parser():
    base_args = dict(
        required=False,
        type=str,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--lang",
        **base_args,
        default="it",
        help="language in [en, nl, de ,it]",
    )
    base_args["help"] = "name of the file to use or full path to it"
    parser.add_argument("-t", "--train", default="gold", **base_args)

    base_args["default"] = "standard"
    parser.add_argument("-dti", "--dev", **base_args)
    parser.add_argument("-tti", "--test", **base_args)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        help="use wandb as a logger",
    )
    parser.add_argument(
        "--wandb_project",
        default="test",
        type=str,
        required=False,
        help="the project to be used with wandb",
    )
    parser.add_argument(
        "--print",
        action=argparse.BooleanOptionalAction,
        help="use base print logger",
    )

    parser.add_argument(
        "--early_stop",
        action=argparse.BooleanOptionalAction,
        help="use early stop mechanism",
    )
    parser.add_argument(
        "-e",
        "--epoch",
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
        required=False,
        default="AdamW",
        help="optimizer class, must belong to torch.optim",
    )

    parser.add_argument(
        "--val_epoch",
        type=int,
        default=4,
        required=False,
        help="Number of train epochs before validating",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-base",
        required=False,
        help="Model name, must belong to some model in the transformers library",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        required=False,
        help="Tokenizer name, if not provided the model tokenizer will be used if available",
    )

    args = parser.parse_args()
    return args
