import argparse
from experiment import ExperimentRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "train", "test"],
        default="all",
        help="all=train+test, train=only train models, test=only run evaluation",
    )
    args = parser.parse_args()

    runner = ExperimentRunner(mode=args.mode)
    if args.mode in ("all", "test"):
        runner.run()


if __name__ == "__main__":
    main()