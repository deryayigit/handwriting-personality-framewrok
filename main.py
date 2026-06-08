import argparse


def list_experiments():
    from src.common.registry import load_experiment_registry

    registry = load_experiment_registry()

    for experiment_id in registry.keys():
        print(experiment_id)


def run():
    parser = argparse.ArgumentParser(
        description="Handwriting Personality Analysis Framework"
    )

    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all registered experiments."
    )

    parser.add_argument(
        "--run-experiment",
        type=str,
        default=None,
        help="Run selected experiment ID."
    )

    parser.add_argument(
        "--eval-classification",
        type=str,
        default=None,
        help="Evaluate an existing classification checkpoint and save confusion matrix."
    )

    parser.add_argument(
        "--eval-dominant-trait",
        type=str,
        default=None,
        help="Evaluate regression output as dominant-trait classification."
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI dashboard."
    )

    args = parser.parse_args()

    if args.list_experiments:
        list_experiments()
        return

    if args.eval_classification:
        from src.classification.evaluate_checkpoint import evaluate_experiment

        evaluate_experiment(args.eval_classification)
        return

    if args.eval_dominant_trait:
        from src.regression.evaluate_dominant_trait import evaluate_regression_dominant_trait

        evaluate_regression_dominant_trait(args.eval_dominant_trait)
        return

    if args.gui:
        from gui.app import main as gui_main

        gui_main()
        return

    if args.run_experiment:
        from src.common.registry import load_experiment_registry
        from src.classification.train import run_classification_experiment
        from src.regression.train import run_regression_experiment

        registry = load_experiment_registry()

        if args.run_experiment not in registry:
            raise ValueError(f"Experiment not found: {args.run_experiment}")

        cfg = registry[args.run_experiment]
        task = cfg.get("task", "classification")

        if task == "classification":
            print(
                f"[INFO] Running classification experiment: "
                f"{args.run_experiment} - {cfg.get('name', '')}"
            )
            run_classification_experiment(cfg)
            return

        if task == "regression":
            print(
                f"[INFO] Running regression experiment: "
                f"{args.run_experiment} - {cfg.get('name', '')}"
            )
            run_regression_experiment(cfg)
            return

        raise ValueError(f"Unknown task type: {task}")

    parser.print_help()


if __name__ == "__main__":
    run()