import os

from udfx.types import STR

from . import Case


def training_image(identifier: str) -> str:
    return os.path.join("dataset", "train", identifier, f"{identifier}.jpg")


def image_identifier(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def image_directory(path: str) -> str:
    return os.path.dirname(path)


def image_parts(path: str) -> str:
    directory, name = os.path.split(path)
    return directory + "|" + name


CASES = [
    Case("training_image_path", training_image, {"identifier": STR}, "supported"),
    Case("image_path_identifier", image_identifier, {"path": STR}, "supported"),
    Case("image_path_directory", image_directory, {"path": STR}, "supported"),
    Case("image_path_parts", image_parts, {"path": STR}, "supported"),
]
