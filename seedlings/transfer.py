"""
Module for training models via transfer learning.
"""
from fire import Fire
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from seedlings.test import make_submission
from seedlings.config import TRAIN_DIR, DEV_DIR
from seedlings.dataset import ImageDataset
from seedlings.utils import RunResult


model_map = {
    "EfficientNetB0": efn.EfficientNetB0,
    "EfficientNetB1": efn.EfficientNetB1,
    "EfficientNetB2": efn.EfficientNetB2,
    "EfficientNetB3": efn.EfficientNetB3,
    "EfficientNetB4": efn.EfficientNetB4,
    "EfficientNetB5": efn.EfficientNetB5,
    "EfficientNetB6": efn.EfficientNetB6,
    "EfficientNetB7": efn.EfficientNetB7,
}


def transfer_train(
    image_size: int = 64,
    batch_size: int = 32,
    epochs: int = 30,
    l2_regularization: float = 1e-5,
    architecture: str = "EfficientNetB0",
    learning_rate: float = 1e-3,
    lr_decay_rate: float = 0.99,
    lr_decay_steps: int = 5e2,
    project_name: str = "Seedlings Image Classification (Transfer Learning)",
) -> RunResult:

    # Load data
    train_data = ImageDataset("train", TRAIN_DIR, target_size=(image_size, image_size))
    print("index2class", train_data.index2class)
    # Add data augmentation to training set.
    train_data_gen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.1).flow(
        train_data.X, train_data.y, batch_size=batch_size
    )
    dev_data = ImageDataset("dev", DEV_DIR, target_size=(image_size, image_size))
    # No data augmentation to dev set.
    dev_data_gen = ImageDataGenerator().flow(
        dev_data.X, dev_data.y, batch_size=batch_size
    )

    # Define and compile the model.
    model = tf.keras.Sequential(
        [
            model_map[architecture](
                input_shape=(image_size, image_size, 3),
                weights="imagenet",
                include_top=False,
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(
                train_data.n_classes,
                activation="softmax",
                kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
            ),
        ]
    )

    model.compile(
        optimizer=Adam(ExponentialDecay(learning_rate, lr_decay_steps, lr_decay_rate)),
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
        validation_data=dev_data_gen,
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
