#!/usr/bin/env python
import logging
import sys

import ase
import ase.io

from compute.ml_layer_finder_engine import process_structure_core
from compute.utils.structures import tuple_from_ase


def run_from_command_line(filename):
    logger = logging.getLogger("tool-ml-layer-finder-tool-app")

    # Print to stderr, also DEBUG messages
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.WARNING)
    logger.addHandler(logging.StreamHandler())

    asecell = ase.io.read(filename)
    structure = tuple_from_ase(asecell)
    return_data = process_structure_core(structure, logger, flask_request=None)
    print("RETURN DATA:", return_data)


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except IndexError:
        print("Pass a filename")
        sys.exit(1)

    run_from_command_line(filename)
