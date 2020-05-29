import os
import json

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import h5py

from seedlings.utils import get_sub_dirs, get_dir_files
from seedlings.utils import make_dir_if_needed


class ImageDataset:
    """
    Loads an image classification dataset.
    """

    # Instance attributes that will get persisted by h5py.
    metadata_fields = [
        "class2index",
        "index2class",
        "classses",
        "n_classes",
    ]

    def __init__(self, name: str, data_root_dir: str, target_size: tuple) -> tuple:
        """
        Parameters
        ----------
        target_size : tuple
            The desired dimensions the image should be resized to (pixels x pixels).
        """
        self.name = name
        self.img_size = target_size
        if os.path.isfile(self.intermediate_path):
            self._from_intermediate()
        else:
            self._from_raw(data_root_dir)
            # Save it in HDF5 format for quicker future use
            self._to_intermediate()

    @property
    def intermediate_path(self) -> str:
        return f"{self.name}-{self.img_size}px.h5"

    def _to_intermediate(self) -> None:
        """
        Saves the dataset to an intermediate HDF5 format,
        for quicker loading in the future.
        """
        h5f = h5py.File(self.intermediate_path, "w")
        h5f.create_dataset("X", data=self.X)
        h5f.create_dataset("y", data=self.y)
        json_metadata_str = json.dumps(
            {field: getattr(self, field) for field in self.metadata_fields}
        )
        h5f.attrs["metadata"] = json_metadata_str
        h5f.close()

    def _from_intermediate(self) -> None:
        print(f"Loading dataset at HDF5 path '{self.intermediate_path}'...")
        h5f = h5py.File(self.intermediate_path, "r")
        self.X = h5f["X"][:]
        self.y = h5f["y"][:]
        json_metadata = json.loads(h5f.attrs["metadata"])
        for field, value in json_metadata.items():
            setattr(self, field, value)

    def _from_raw(self, data_root_dir: str) -> None:
        self.class2index = {}
        self.index2class = {}
        self.classses = []
        X, y = [], []

        print(f"Loading dataset at root path: '{data_root_dir}'...")
        for i, class_path in enumerate(tqdm(get_sub_dirs(data_root_dir))):
            class_name = class_path.split("/")[-1]
            self.class2index[class_name] = i
            self.index2class[i] = class_name
            self.classses.append(class_name)

            for image_fname in get_dir_files(class_path):
                image = load_img(
                    os.path.join(class_path, image_fname), target_size=self.img_size
                )
                image_arr = img_to_array(image)
                X.append(image_arr)
                y.append(i)

        self.X = np.array(X) / 255.0  # normalize the scale
        self.y = to_categorical(y)
        self.n_classes = len(self.classses)

