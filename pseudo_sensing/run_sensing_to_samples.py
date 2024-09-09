import os
import argparse

from tqdm import tqdm

from modules.pseudo_sensing import run_sensing
from modules.utils import load_samples_json

def run(test_data, sample, args):
    """Run pseudo sensing on a sample.

    Args:
    sample: dict, sample data
    args: argparse.Namespace, arguments
    """
