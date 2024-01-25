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
        default="en",
        help="language in [en, nl, de ,it]",
    )
    base_args["help"] = "name of the file to use or full path to it"
    parser.add_argument("-t", "--train", default="gold", **base_args)

    base_args["default"] = "standard"
    parser.add_argument("-dti", "--dev", **base_args)
    parser.add_argument("-tti", "--test", **base_args)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--print", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args
