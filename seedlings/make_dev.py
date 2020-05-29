"""
Makes the development dataset by taking 30% randomly selected, stratified
images out of the `test` images.
"""
import os
import random
from pprint import pprint

from fire import Fire

from seedlings.config import TRAIN_DIR, DEV_DIR
from seedlings.utils import get_sub_dirs, make_dir_if_needed, get_dir_files

DEV_RATIO = 0.3


def report_dataset_stats(dataset_path: str, dataset_name: str) -> None:
    class_paths = get_sub_dirs(dataset_path)
    stats = {
        class_dir.split("/")[-1]: len(get_dir_files(class_dir))
        for class_dir in class_paths
    }
    print(f"{dataset_name} Set:")
    pprint(stats, indent=4)


def move_all_to_train() -> None:
    """
    Moves all the dev images back to the train set.
    """
    class_paths = get_sub_dirs(DEV_DIR)
    classes = [p.split("/")[-1] for p in class_paths]
    for i, class_path in enumerate(class_paths):
        # Get the file paths of all images for this class.
        images = get_dir_files(class_path)
        train_class_path = os.path.join(TRAIN_DIR, classes[i])

        for image in images:
            # Move this image to the train set.
            os.replace(
                os.path.join(class_path, image), os.path.join(train_class_path, image),
            )


def make_dev_set() -> None:
    make_dir_if_needed(DEV_DIR)
    class_paths = get_sub_dirs(TRAIN_DIR)
    classes = [p.split("/")[-1] for p in class_paths]

    for i, class_path in enumerate(class_paths):
        # Get the file paths of all images for this class.
        images = get_dir_files(class_path)

        n_images = len(images)
        n_dev = int(n_images * DEV_RATIO)
        print(f"The {class_path} class will have {n_dev}/{n_images} dev images")

        # Randomly sample `n_dev` images for the dev set for this class.
        dev_images = random.sample(images, n_dev)
        dev_class_path = os.path.join(DEV_DIR, classes[i])
        make_dir_if_needed(dev_class_path)

        for dev_image in dev_images:
            # Move this image to the dev set.
            os.replace(
                os.path.join(class_path, dev_image),
                os.path.join(dev_class_path, dev_image),
            )


def main(*, make_dev: bool = False, reset: bool = False) -> None:
    if make_dev:
        make_dev_set()

    if reset:
        move_all_to_train()

    report_dataset_stats(TRAIN_DIR, "Train")
    report_dataset_stats(DEV_DIR, "Dev")


if __name__ == "__main__":
    Fire(main)
