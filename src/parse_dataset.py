from typing import List, Tuple, Dict
import cv2
from pathlib import Path
import numpy as np
import logging
import pickle
from frozendict import frozendict

invisible_eye_geom = frozendict({'screen_width_mm': 641.3,
                                 'screen_height_mm': 400.8,
                                 'screen_width_pix': 2560.,
                                 'screen_height_pix': 1600.,
                                 'eye_distance_mm': 510.})


def gaze_pix2deg(pos: np.array, geom: dict = invisible_eye_geom) -> np.array:
    assert len(pos.shape) == 2 and pos.shape[1] == 2

    new_pos = np.zeros(pos.shape)

    ratio_x = geom['screen_width_mm'] / geom['screen_width_pix']
    ratio_y = geom['screen_height_mm'] / geom['screen_height_pix']
    # shift and scale to move from pixel coordinate system
    # to the metric coordinate system (mm) with the center of the screen at (0, 0)
    # note: screen center (0, 0) is not always a neutral eye position (for invisible eye dataset it's not)
    new_pos[:, 0] = (pos[:, 0] - geom['screen_width_pix'] / 2) * ratio_x
    new_pos[:, 1] = (pos[:, 1] - geom['screen_height_pix'] / 2) * ratio_y
    # calculate degrees of visual angle
    new_pos[:, 0] = np.rad2deg(np.arctan2(new_pos[:, 0], geom['eye_distance_mm']))
    new_pos[:, 1] = np.rad2deg(np.arctan2(new_pos[:, 1], geom['eye_distance_mm']))

    return new_pos


def preprocess_image(img: np.array, resize_shape: Tuple[int, int]):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[
          img.shape[0] // 4: img.shape[0] * 3 // 4, img.shape[1] // 4: img.shape[1] * 3 // 4
          ]
    img = cv2.resize(img, resize_shape)
    return img


def parse_invisible_eye_videostream(path: Path, img_filename_pattern: str, n_images: int,
                                    resize_shapes: List[Tuple[int, int]]) -> Dict[Tuple[int, int], np.array]:
    result = {resize_shape: np.zeros((n_images,) + resize_shape) for resize_shape in resize_shapes}
    images_paths = set(path.glob(img_filename_pattern))

    if len(images_paths) != n_images:
        raise ValueError("Number of images should be the same as the number of known eye positions")

    for img_path in images_paths:
        img = cv2.imread(str(img_path))
        img_id = int(img_path.stem)

        if img_id >= n_images:
            raise ValueError("Image id (name) should be < then number of images")

        for resize_shape in resize_shapes:
            result[resize_shape][img_id] = preprocess_image(img, resize_shape)
    return result


# (Dict)[resize_shape](List)[subject_id](np.array)[camera_id, image_id, img_x, img_y]
def parse_invisible_eye_dataset(dataset_path: Path, resize_shapes: List[Tuple[int, int]]) -> \
        Tuple[Dict[Tuple[int, int], List[np.array]], List[np.array],
              Dict[Tuple[int, int], List[np.array]], List[np.array]]:
    n_cameras = 4
    x_train: Dict[Tuple[int, int], List[np.array]] = {resize_shape: [] for resize_shape in resize_shapes}
    x_test: Dict[Tuple[int, int], List[np.array]] = {resize_shape: [] for resize_shape in resize_shapes}
    y_train: List[np.array] = []
    y_test: List[np.array] = []

    for subject_id, subject_path in enumerate(sorted(dataset_path.glob("*"))):
        logging.info("Processing subject {}".format(subject_path.name))
        train_path = subject_path / 'train'
        test_path = subject_path / 'test'

        y_train.append(gaze_pix2deg(np.load(train_path / 'Y.npy')))
        y_test.append(gaze_pix2deg(np.load(test_path / 'Y.npy')))

        for resize_shape in resize_shapes:
            x_train[resize_shape].append(np.zeros((n_cameras, y_train[-1].shape[0],) + resize_shape))
        for resize_shape in resize_shapes:
            x_test[resize_shape].append(np.zeros((n_cameras, y_test[-1].shape[0],) + resize_shape))
        logging.info("N train: {}, N test: {}".format(y_train[-1].shape[0], y_test[-1].shape[0]))

        for camera_id in range(n_cameras):
            train_entry = parse_invisible_eye_videostream(train_path.joinpath(str(camera_id)), "*.png",
                                                          y_train[-1].shape[0], resize_shapes)
            test_entry = parse_invisible_eye_videostream(test_path.joinpath(str(camera_id)), "*.png",
                                                         y_test[-1].shape[0], resize_shapes)
            logging.debug("Camera {}. N train: {}, N test: {}".format(
                camera_id, train_entry[(1, 1)].shape[0], test_entry[(1, 1)].shape[0])
            )
            for resize_shape, videostream in train_entry.items():
                x_train[resize_shape][subject_id][camera_id] = videostream

            for resize_shape, videostream in test_entry.items():
                x_test[resize_shape][subject_id][camera_id] = videostream

    return x_train, y_train, x_test, y_test


def load_invisible_eye_dataset(dataset_path: Path) -> \
        Tuple[List[np.array], List[np.array], List[np.array], List[np.array]]:
    with open(str(dataset_path), 'rb') as f:
        data_dict = pickle.load(f)

    return data_dict['x_train'], data_dict['y_train'], data_dict['x_test'], data_dict['y_test']


def save_invisible_eye_dataset(x_train: List[np.array], y_train: List[np.array],
                               x_test: List[np.array], y_test: List[np.array], path: Path) -> None:
    # extensive check to avoid saving errors
    assert len(x_train[0].shape) == len(x_test[0].shape) == 4 and len(y_train[0].shape) == len(y_test[0].shape) == 2
    assert len(x_train) == len(y_train) == len(x_test) == len(y_test)
    assert all([x_train[i].shape[0] == x_test[i].shape[0] == 4 for i in range(len(x_train))])
    assert all([x_train[i].shape[2] == x_train[i].shape[3] == x_test[i].shape[2] == x_test[i].shape[3]
                for i in range(len(x_train))])
    assert all([y_train[i].shape[1] == y_test[i].shape[1] == 2 for i in range(len(y_train))])
    assert all([x_train[i].shape[1] == y_train[i].shape[0] for i in range(len(x_train))])
    assert all([x_test[i].shape[1] == y_test[i].shape[0] for i in range(len(x_test))])
    # Sanity check if train set is bigger then the test set. It shouldn't always be true, but for this dataset it is.
    assert all([x_train[i].shape[1] > x_test[i].shape[1] for i in range(len(x_train))])

    dataset = {'x_train': x_train, 'y_train': y_train,
               'x_test': x_test, 'y_test': y_test}
    with open(str(path), 'wb') as f:
        pickle.dump(dataset, f)
