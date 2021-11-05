#!/usr/bin/env python3

from typing import NamedTuple, List, Union, Tuple


from numpy import ndarray
import json
import os.path
from console.main import SimilarityCalculator
from pathlib import Path
import numpy as np


def _deserialize_file(pathname: Union[str, Path]):
    with open(pathname, 'r') as ifile:
        return json.load(ifile)


class CalcCase(NamedTuple):

    embedding1: ndarray
    embedding2: ndarray
    expected: float
    filenames: Tuple[str, str]

    @staticmethod
    def from_object(obj: list, filenames: Tuple[str, str]) -> 'CalcCase':
        embedding1 = np.array(obj[0])
        embedding2 = np.array(obj[1])
        score = obj[2]
        return CalcCase(embedding1, embedding2, score, filenames)

    @staticmethod
    def load_all() -> List['CalcCase']:
        test_case_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'embedding-classifier-test-cases'
        distances = _deserialize_file(test_case_dir / 'distances.json')
        test_cases = []
        for filename1, filename2, score in distances:
            embedding1 = _deserialize_file(test_case_dir / filename1)
            embedding2 = _deserialize_file(test_case_dir / filename2)
            test_case = CalcCase.from_object([embedding1, embedding2, score], (filename1, filename2))
            test_cases.append(test_case)
        return test_cases


_THRESH = 0.0001

def test_calculate():
    test_cases = CalcCase.load_all()
    c = SimilarityCalculator()
    for test_case in test_cases:
        score = c.calculate(test_case.embedding1, test_case.embedding2)
        print(test_case.filenames, score)
        delta = abs(test_case.expected - score)
        assert delta <= _THRESH, f"score {score} expected {test_case.expected} for {test_case.filenames}"




