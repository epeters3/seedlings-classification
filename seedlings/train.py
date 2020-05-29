"""
Training the model.
"""
from typing import NamedTuple

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from fire import Fire
import wandb
from wandb.keras import WandbCallback
import pandas as pd

from seedlings.config import TRAIN_DIR, DEV_DIR
from seedlings.model import make_cnn
from seedlings.dataset import ImageDataset
from seedlings.test import make_submission


class RunResult(NamedTuple):
    dev_acc: float
    train_acc: float
    run_name: str
    submission: pd.DataFrame


def train(
    *,
    image_size: int = 64,
    batch_size: int = 32,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    lr_decay_rate: float = 0.99,
    lr_decay_steps: int = 5e2,
    project_name: str = "Seedlings Image Classification",
    **model_params,
) -> RunResult:
    """
    Used to build and train a CNN model on the seedlings dataset, according to the
    specified parameters. Also tracks the parameters, training progress, and final
    performance scores, and reports them all to the `wandb` service.

    Parameters
    ----------
    image_size : int
        The dimension to use for the height and width of the image during training.
    batch_size : int
        The batch size to use during training and evaluation.
    epochs : int
        The number of epochs to train the model for.
    **model_params
        All other keyword arguments are forwarded on to the `seedlings.model.make_cnn` method.
    """

    # Load data
    train_data = ImageDataset("train", TRAIN_DIR, target_size=(image_size, image_size))
    # Add data augmentation to training set.
    train_data_gen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.1).flow(
        train_data.X, train_data.y, batch_size=batch_size
    )
    dev_data = ImageDataset("dev", DEV_DIR, target_size=(image_size, image_size))
    # No data augmentation to dev set.
    dev_data_gen = ImageDataGenerator().flow(
        dev_data.X, dev_data.y, batch_size=batch_size
    )

    # Make and init the wandb run.
    wandb.init(project=project_name, reinit=True)
    wandb.config.update(
        {
            "batch_size": batch_size,
            "epochs": epochs,
            "image_size": image_size,
            "learning_rate": learning_rate,
            "lr_decay_rate": lr_decay_rate,
            "lr_decay_steps": lr_decay_steps,
        }
    )

    # Make and compile the model with given hyperparameters.
    model = make_cnn(
        image_size=image_size, n_classes=train_data.n_classes, **model_params,
    )
    model.compile(
        optimizer=Adam(ExponentialDecay(learning_rate, lr_decay_steps, lr_decay_rate)),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    wandb.config.num_params = model.count_params()

    # Train the model.
    print(f"Training the model...")
    model.fit(
        train_data_gen,
        steps_per_epoch=int(len(train_data.X) / batch_size),
        epochs=epochs,
        validation_data=dev_data_gen,
        callbacks=[WandbCallback(save_model=False)],
    )

    # Evaluate final performance on the train set
    train_loss, train_acc = model.evaluate(train_data.X, train_data.y)

    # Evaluate final performance on the dev set
    dev_loss, dev_acc = model.evaluate(dev_data.X, dev_data.y)

    # Log the scores
    wandb.run.summary.update(
        {
            "final_val_loss": dev_loss,
            "final_val_acc": dev_acc,
            "final_train_loss": train_loss,
            "final_train_acc": train_acc,
        }
    )
    wandb.run.save()
    run_name = wandb.run.name
    wandb.join()  # end this run

    return RunResult(
        dev_acc,
        train_acc,
        run_name,
        make_submission(model, image_size, train_data.index2class),
    )


if __name__ == "__main__":
    Fire(train)
