"""
Module for training models via transfer learning.
"""
from fire import Fire
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow_hub as hub

from seedlings.test import make_submission
from seedlings.config import TRAIN_DIR, DEV_DIR
from seedlings.dataset import ImageDataset
from seedlings.utils import RunResult


model_map = {"BiT-M R50x1": "https://tfhub.dev/google/bit/m-r50x1/1"}


def transfer_train(
    image_size: int = 128,
    batch_size: int = 32,
    epochs: int = 30,
    l2_regularization: float = 1e-5,
    architecture: str = "BiT-M R50x1",
    learning_rate: float = 1e-3,
    lr_decay_rate: float = 0.99,
    lr_decay_steps: int = 5e2,
    project_name: str = "Seedlings Image Classification (Transfer Learning)",
) -> RunResult:

    # Load data
    train_data = ImageDataset("train", TRAIN_DIR, target_size=(image_size, image_size))
    # Add data augmentation to training set.
    train_data_gen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.1).flow(
        train_data.X, train_data.y, batch_size=batch_size
    )
    dev_data = ImageDataset("dev", DEV_DIR, target_size=(image_size, image_size))

    # Define and compile the model.
    model = tf.keras.Sequential(
        [
            hub.KerasLayer(model_map[architecture], trainable=False),
            L.Flatten(),
            L.Dense(
                train_data.n_classes,
                activation="softmax",
                kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
            ),
        ]
    )

    model.compile(
        optimizer=Adam(
            ExponentialDecay(
                learning_rate, lr_decay_steps, lr_decay_rate, staircase=True
            )
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Make and init the wandb run.
    wandb.init(project=project_name, reinit=True)
    wandb.config.update(
        {
            "batch_size": batch_size,
            "epochs": epochs,
            "image_size": image_size,
            "l2_regularization": l2_regularization,
            "architecture": architecture,
            "learning_rate": learning_rate,
            "lr_decay_rate": lr_decay_rate,
            "lr_decay_steps": lr_decay_steps,
        }
    )

    # Train the model
    model.fit(
        train_data_gen,
        steps_per_epoch=int(len(train_data.X) / batch_size),
        epochs=epochs,
        validation_data=(dev_data.X, dev_data.y),
        callbacks=[WandbCallback(save_model=False)],
    )

    # Evaluate the model
    train_loss, train_acc = model.evaluate(train_data.X, train_data.y)
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
    Fire(transfer_train)
