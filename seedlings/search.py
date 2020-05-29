"""
Performing hyperparameter search.
"""
from fire import Fire
import random
from pprint import pprint

from seedlings.train import train
from seedlings.test import save_submission


def make_discrete_sampler(options):
    return lambda: random.choice(list(options))


def make_int_sampler(lbound: int, ubound: int):
    return lambda: random.randint(lbound, ubound)


def make_sampler(lbound: float, ubound: float, scale: str = "uniform", base: int = 10):
    """
    If `scale=="uniform"`, returns a function that when called samples
    uniformly from the range `[lbound, ubound]`. When `scale=="log"`,
    returns a function that samples from a log scale with the power
    being drawn uniformly from the range `[lbound, ubound]`. E.g.
    `make_sampler(-4,-1,"log")()` will return a number between
    `1e-4` and `1e-1` drawn unifromly from a base 10 log scale.
    """
    if scale == "uniform":
        return lambda: random.uniform(lbound, ubound)
    elif scale == "log":
        return lambda: base ** random.uniform(lbound, ubound)
    else:
        raise ValueError("unsupported sample scale")


hyperparam_samplers = {
    "epochs": make_int_sampler(50, 200),
    "learning_rate": make_sampler(-6, -1, "log"),
    "lr_decay_rate": make_sampler(0.8, 1.0),
    "lr_decay_steps": make_sampler(1e1, 1e4),
    "l2_regularization": make_sampler(-10, -1, "log"),
    "dropout_rate": make_sampler(0, 0.8),
    "kernel_size": make_discrete_sampler({3, 5, 7, 9}),
    "initial_filters": make_int_sampler(16, 128),
    "max_filters": make_int_sampler(32, 128),
}


def sample_params() -> dict:
    return {
        param_name: sampler() for param_name, sampler in hyperparam_samplers.items()
    }


def search(*, nsamples: int, nsave: int) -> dict:
    """
    Randomly samples `nsamples` models and trains them on the
    training data, making and saving a kaggle submission for
    each of the `nsave` best models, best meaning the ones with
    the maximum dev set accuracy. Returns the history of the runs.
    """
    # Sample and train `nsamples` different models.
    samples = []
    for _ in range(nsamples):
        params = sample_params()
        print("training model with params:")
        pprint(params)
        run_result = train(
            **params, project_name="Seedlings Image Classification (Search)"
        )
        samples.append(run_result)

    # Find and save submissions for the `nsave` best ones.
    samples.sort(key=lambda result: result.dev_acc, reverse=True)
    nbest = samples[:nsave]
    for rank, result in enumerate(nbest, 1):
        save_submission(result.submission, f"search-rank-{rank}--{result.run_name}")


if __name__ == "__main__":
    Fire(search)
