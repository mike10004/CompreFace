#!/usr/bin/env python3

from sample_images import IMG_DIR
import glob
from console import main

def _load_image_pathnames(filename_pattern:str=None):
    return glob.glob(str(IMG_DIR / (filename_pattern or "*.jpg")))


def test_detect():
    scanner = main.MainFaceScanner()
    mainer = main.Mainer(scanner)
    image_pathnames = _load_image_pathnames("00[567]_*.jpg")
    assert image_pathnames
    detections = mainer.scan(image_pathnames)
    assert 4 == len(detections)
    print("hello, this test has passed")
