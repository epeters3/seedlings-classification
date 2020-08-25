"""
Performing hyperparameter search.
"""
from fire import Fire
import random
from pprint import pprint

from seedlings.train import train
from seedlings.test import save_submission
from seedlings.transfer import transfer_train


def make_discrete_sampler(options):
    return lambda: random.choice(list(options))


def make_sampler(lbound, ubound, method: str = "uniform", base: int = 10):
    """
    If `method=="uniform"`, returns a function that when called samples
    uniformly from the range `[lbound, ubound]`. When `method=="log"`,
    returns a function that samples from a log method with the power
    being drawn uniformly from the range `[lbound, ubound]`. E.g.
    `make_sampler(-4,-1,"log")()` will return a number between
    `1e-4` and `1e-1` drawn unifromly from a base 10 log method. When
    `method=="int"`, returns a function that when called samples an
    integer uniformly from the range `[lbound, ubound]`.
    """
    if method == "uniform":
        return lambda: random.uniform(lbound, ubound)
    elif method == "log":
        return lambda: base ** random.uniform(lbound, ubound)
    elif method == "int":
        return lambda: random.randint(lbound, ubound)
    else:
        raise ValueError("unsupported sample method")


# Hyperparams for `seedlings.train.train`
scratch_hyperparam_samplers = {
    "epochs": make_sampler(50, 200, "int"),
    "learning_rate": make_sampler(-6, -1, "log"),
    "lr_decay_rate": make_sampler(0.8, 1.0),
    "lr_decay_steps": make_sampler(1e1, 1e4),
    "l2_regularization": make_sampler(-10, -1, "log"),
    "dropout_rate": make_sampler(0, 0.8),
    "kernel_size": make_discrete_sampler({3, 5, 7, 9}),
    "initial_filters": make_sampler(16, 128, "int"),
    "max_filters": make_sampler(32, 128, "int"),
}

# Hyperparams for `seedlings.transfer.transfer_train`
pretrained_hyperparam_samplers = {
    "epochs": make_sampler(5, 35, "int"),
    "l2_regularization": make_sampler(-10, -1, "log"),
    "architecture": make_discrete_sampler({"BiT-M R50x1"}),
    "learning_rate": make_sampler(-6, -1, "log"),
    "lr_decay_rate": make_sampler(0.8, 1.0),
    "lr_decay_steps": make_sampler(1e1, 1e4),
}


def sample_params(hyperparam_samplers: dict) -> dict:
    return {
        param_name: sampler() for param_name, sampler in hyperparam_samplers.items()
    }


def search(*, nsamples: int, nsave: int, search_space: str) -> dict:
    """
    Randomly samples `nsamples` models and trains them on the
    training data, making and saving a kaggle submission for
    each of the `nsave` best models, best meaning the ones with
    the maximum dev set accuracy. Returns the history of the runs.
    `search_space` specifies whether to use from scratch models
    or pretrained.
    """
    # Configure based on search space
    if search_space == "scratch":
        project_name = "Seedlings Image Classification (Search)"
        hyperparam_samplers = scratch_hyperparam_samplers
        train_method = train
    elif search_space == "pretrained":
        project_name = "Seedlings Image Classification (Transfer Search)"
        hyperparam_samplers = pretrained_hyperparam_samplers
        train_method = transfer_train
    else:
        raise ValueError("unsupported search space")

    # Sample and train `nsamples` different models.
    samples = []
    for _ in range(nsamples):
        params = sample_params(hyperparam_samplers)
        print("training model with params:")
        pprint(params)
        run_result = train_method(**params, project_name=project_name)
        samples.append(run_result)

    # Find and save submissions for the `nsave` best ones.
    samples.sort(key=lambda result: result.dev_acc, reverse=True)
    nbest = samples[:nsave]
    for rank, result in enumerate(nbest, 1):
        save_submission(result.submission, f"search-rank-{rank}--{result.run_name}")


if __name__ == "__main__":
    Fire(search)
