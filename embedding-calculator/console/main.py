#!/usr/bin/env python3

import os.path
import sys
from argparse import ArgumentParser
import logging

from numpy import ndarray
import numpy as np

from src.services.dto.bounding_box import BoundingBoxDTO
from src.services.facescan.plugins import mixins
from src.services.facescan.scanner.facescanner import FaceScanner
from src.services.imgtools.read_img import read_img
from typing import TextIO, List, NamedTuple, Tuple, Sequence, Callable
from src.services.dto.plugin_result import FaceDTO
from src.services.imgtools.types import Array3D


_log = logging.getLogger(__name__)


class Detection(NamedTuple):

    image_file: str
    index: int
    face: FaceDTO


# noinspection PyMethodMayBeStatic
class SimilarityCalculator(object):

    def __init__(self, similarity_coefficients: Tuple[float, float]=None):
        self.similarity_coefficients = similarity_coefficients or (1.1817961, 5.291995557)

    def normalize(self, embeddings: ndarray) -> ndarray:
        norm_val = np.linalg.norm(embeddings, ord=2)
        return embeddings / norm_val

    def to_similarity(self, distance: ndarray) -> ndarray:
        c0, c1 = self.similarity_coefficients
        intermed = (c0 - distance) * c1
        tformed = np.tanh(intermed)
        return (tformed + 1) / 2

    def calculate(self, array1: ndarray, array2: ndarray) -> float:
        norm_array1 = self.normalize(array1)
        norm_array2 = self.normalize(array2)
        norm_sub = norm_array1 - norm_array2
        eucl_dist = np.linalg.norm(norm_sub, ord=2)
        sim = self.to_similarity(eucl_dist)
        return float(sim)


class Mainer(object):

    def __init__(self, scanner: FaceScanner, stdout: TextIO = sys.stdout, detection_threshold: float = 0.5):
        self.scanner = scanner
        self.stdout = stdout
        self.detection_threshold = detection_threshold
        self.similarity_calculator = SimilarityCalculator()
        self.abs_paths = False

    def scan(self, image_files: List[str]) -> List[Detection]:
        detections = []
        for image_file in image_files:
            try:
                faces = self.scanner.scan(
                    img=read_img(image_file),
                    det_prob_threshold=self.detection_threshold
                )
                current_detections = [Detection(image_file, index, face) for index, face in enumerate(faces)]
                detections += current_detections
                for detection in current_detections:
                    box = [detection.face.box.x_min, detection.face.box.y_min, detection.face.box.x_max, detection.face.box.y_max, ]
                    print(os.path.split(image_file)[1], detection.index, box, file=self.stdout)
            except IOError as e:
                _log.warning("detection failed on %s: %s", image_file, str(e))
        return detections

    def _compute_distance(self, embedding1: ndarray, embedding2: ndarray):
        return self.similarity_calculator.calculate(embedding1, embedding2)

    def _to_path(self, detection: Detection):
        p = detection.image_file
        return p if self.abs_paths else os.path.split(p)[1]

    def compare(self, detections: List[Detection]):
        for i in range(len(detections)):
            d1 = detections[i]
            d1_embedding = d1.face.embedding
            path1 = self._to_path(d1)
            for j in range(i + 1, len(detections)):
                d2 = detections[j]
                d2_embedding = d2.face.embedding
                dist = self._compute_distance(d1_embedding, d2_embedding)
                path2 = self._to_path(d2)
                print(dist, path1, d1.index, path2, d2.index, file=self.stdout)


class MainFaceScanner(FaceScanner):
    """
    The scanner only performs face detection and embedding calculation.
    """

    @property
    def difference_threshold(self) -> float:
        raise NotImplementedError("not supported by this implementation")

    ID = "MainFaceScanner"

    def __init__(self):
        super().__init__()
        import src.services.facescan.plugins.facenet.facenet
        self.detector: mixins.FaceDetectorMixin = src.services.facescan.plugins.facenet.facenet.FaceDetector() # mixins.FaceDetectorMixin = [pl for pl in plugins if isinstance(pl, mixins.FaceDetectorMixin)][0]
        self.calculator: mixins.CalculatorMixin = src.services.facescan.plugins.facenet.facenet.Calculator() # [pl for pl in plugins if isinstance(pl, mixins.CalculatorMixin)][0]

    def scan(self, img: Array3D, det_prob_threshold: float = None) -> List[FaceDTO]:
        # noinspection PyTypeChecker
        faces = self.detector(img, det_prob_threshold, [self.calculator])
        return faces

    def find_faces(self, img: Array3D, det_prob_threshold: float = None) -> List[BoundingBoxDTO]:
        return self.detector.find_faces(img, det_prob_threshold)


def _disable_cuda():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _os_setenv(var_name: str, var_value: str):
    os.environ[var_name] = var_value


def _set_if_unset(var_name: str,
                  var_value: str,
                  getter: Callable[[str], str]=os.getenv,
                  setter: Callable[[str, str], None]=_os_setenv):
    current_value = None
    try:
        current_value = getter(var_name)
    except KeyError:
        pass
    if current_value is None:
        setter(var_name, var_value)


def _is_integer(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def _parse_image_spec(image_files: List[str]) -> List[str]:
    if not image_files:
        raise ValueError("image files must be specified")
    if image_files[0] == 'samples':
        limit = None
        pattern = "*.*"
        for token in image_files[1:]:
            if _is_integer(token):
                limit = int(token)
            else:
                pattern = token
        import glob
        from sample_images import IMG_DIR
        images = sorted(glob.glob(str(IMG_DIR / pattern)))
        if limit is not None:
            return images[:limit]
        else:
            return images
    return image_files


def main(argv1: Sequence[str]=None):
    parser = ArgumentParser()
    parser.add_argument("images", metavar='FILE', nargs='+', help="image files to scan/compare")
    parser.add_argument("-l", "--log-level", metavar='LEVEL', choices=('debug', 'info', 'warn', 'error'), default='info', help="set log level (debug, info, warn, or error)")
    parser.add_argument("--tf-logging", choices=('quiet', 'verbose'), help="tensorflow logging mode ('quiet' or 'verbose')")
    parser.add_argument("--disable-cuda", action='store_true', help="set environment variable that indcates zero GPUs")
    args = parser.parse_args(argv1)
    logging.basicConfig(level=logging.__dict__[args.log_level.upper()])
    if args.tf_logging == 'quiet':
        _set_if_unset('TF_CPP_MIN_LOG_LEVEL', '3')
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
    if args.disable_cuda:
        _disable_cuda()
    scanner = MainFaceScanner()
    mainer = Mainer(scanner)
    image_files = _parse_image_spec(args.images)
    detections = mainer.scan(image_files)
    mainer.compare(detections)
    return 0


if __name__ == '__main__':
    exit(main())
