import os

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import h5py
import pandas as pd

from seedlings.utils import get_dir_files, make_dir_if_needed


def get_test(image_size: tuple, test_dir: str) -> None:
    h5path = f"test-{image_size}px.h5"

    if os.path.isfile(h5path):
        h5f = h5py.File(h5path, "r")
        X = h5f["X"][:]
    else:
        X = []

        fnames = sorted(get_dir_files(test_dir))
        for image_fname in fnames:
            image = load_img(
                os.path.join(test_dir, image_fname), target_size=image_size
            )
            image_arr = img_to_array(image)
            X.append(image_arr)

        X = np.array(X) / 255.0  # normalize the scale

        h5f = h5py.File(h5path, "w")
        h5f.create_dataset("X", data=X)
        h5f.close()

    return X


def make_submission(model, image_size: int, index2class: dict) -> pd.DataFrame:
    test_X = get_test((image_size, image_size), "./test")
    test_probs = model.predict(test_X)
    test_preds = test_probs.argmax(axis=-1)
    test_fnames = sorted(get_dir_files("./test"))
    test_labels = []
    for pred in test_preds:
        test_labels.append(index2class[str(pred)])
    return pd.DataFrame({"file": test_fnames, "species": test_labels})


def save_submission(submission: pd.DataFrame, name: str) -> None:
    make_dir_if_needed("submissions")
    submission.to_csv(f"submissions/{name}.csv", index=False)


def make_and_save_submission(
    model, *, image_size: int, index2class: dict, name: str
) -> None:
    submission = make_submission(model, image_size, index2class)
    save_submission(submission, name)
