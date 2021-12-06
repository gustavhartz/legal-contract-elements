# -*- coding: utf-8 -*-
"""Download and pre-process SQuAD and GloVe datasets
Usage:
    > source activate squad
    > python setup.py
Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py
    > https://github.com/chrischute/squad
Author:
    Gustav Hartz (s174315@student.dtu.dk)
"""
import logging
import os
import urllib.request
import warnings
from codecs import open
from collections import Counter
from pathlib import Path
from subprocess import run
from zipfile import ZipFile

import numpy as np
from args_cuad import get_setup_args
from helpers_cuad import data_frame_from_cuda
from tqdm import tqdm

from src.utils.log import get_logger

logger = get_logger(__name__)
logger_spacy = logging.getLogger("spacy")
logger_spacy.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"\[W108\]", category=UserWarning)


def main():
    logger.info('Fetching and processing dataset')
    args_ = get_setup_args()
    f_loc = download(args_)
    logger.info('Data fetched')
    logger.info('Processing raw data')
    """
    Ensure files are there

    Process them
    """
    data_frame_from_cuda(f_loc.get("CUDA V1") + "/CUAD_v1/CUAD_v1.json")


def download(args):
    downloads = [
        # All the data elements we wish to download
        ('GloVe word vectors', args.glove_url),
        ('CUDA V1', args.data_url)
    ]
    file_locations = {}
    for name, url in downloads:
        output_path = str(Path(__file__).resolve().parents[2]) + "/data/raw/" + url.split("/")[-1].split("?")[0]
        file_locations[name] = output_path
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)
            file_locations[name] = extracted_path

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    return file_locations


def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    main()
