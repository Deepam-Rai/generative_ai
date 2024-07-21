import os
from pathlib import Path
from typing import List
import cchardet
from pypdf import PdfReader
import logging


logger = logging.getLogger(__name__)


def get_files(file_path: Path) -> List[Path]:
    """
    Given a file-path, if it is a directory, recursively iterates to get list of files at leaf level.
    :param file_path:
    :return:
    """
    if os.path.isfile(file_path):
        return [file_path]
    elif os.path.isdir(file_path):
        paths = []
        for item in os.listdir(file_path):
            paths += get_files(file_path/item)
        return paths


def get_text(file_paths: List[Path]) -> str:
    """
    :param file_paths: list of filepaths
    :return: extracted text
    """
    text = ""
    files = []
    for path in file_paths:
        files += get_files(path)
    logger.info(f"Extracting...")
    for file_name in files:
        logger.info(f"\t{file_name}")
        _, file_extension = os.path.splitext(file_name)
        if file_extension.lower() == '.pdf':
            reader = PdfReader(file_name)
            for page in reader.pages:
                text += ' ' + page.extract_text()
        elif file_extension.lower() == '.txt':
            blob = file_name.read_bytes()
            detection = cchardet.detect(blob)
            encoding = detection["encoding"]
            with open(file_name, 'r', encoding=encoding) as f:
                text += ' ' + f.read()
        else:
            raise ValueError(f"Unknown filetype: {file_name}")
    return text