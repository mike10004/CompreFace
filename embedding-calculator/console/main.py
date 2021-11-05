#!/usr/bin/env python3

import os.path
import sys
from argparse import ArgumentParser
import logging

from numpy import ndarray
import numpy as np

from src.services.imgtools.read_img import read_img
from typing import TextIO, List, NamedTuple, Tuple
from src.services.facescan.scanner.facescanners import scanner
from src.services.dto.plugin_result import FaceDTO

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

    def __init__(self, stdout: TextIO = sys.stdout, detection_threshold: float = 0.5):
        self.stdout = stdout
        self.detection_threshold = detection_threshold
        self.similarity_calculator = SimilarityCalculator()
        self.abs_paths = False

    def scan(self, image_files: List[str]) -> List[Detection]:
        detections = []
        for image_file in image_files:
            try:
                faces = scanner.scan(
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


def main():
    parser = ArgumentParser()
    parser.add_argument("images", metavar='FILE [FILES...]', nargs='+', help="image files to scan/compare")
    parser.add_argument("-l", "--log-level", choices=('debug', 'info', 'warn', 'error'), default='info', help="set log level")
    args = parser.parse_args()
    logging.basicConfig(level=logging.__dict__[args.log_level.upper()])
    mainer = Mainer()
    detections = mainer.scan(args.images)
    mainer.compare(detections)
    return 0


if __name__ == '__main__':
    exit(main())
