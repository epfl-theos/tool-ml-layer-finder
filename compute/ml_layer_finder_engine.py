import json
import math
import time
import string

import ase
import ase.io
import numpy as np
from ase.data import chemical_symbols
import spglib

from collections.abc import Iterable

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from spglib.spglib import get_symmetry_dataset

from .utils.structures import (
    ase_from_tuple,
    get_xsf_structure,
    tuple_from_ase,
    get_covalent_radii_array,
)
from tools_barebone import get_tools_barebone_version
from .utils.hall import hall_numbers_of_spacegroup
from .utils.layers import find_layers, find_common_transformation
from .utils.pointgroup import (
    pg_number_from_hm_symbol,
    prepare_pointgroup,
    prepare_spacegroup,
    SYMPREC,
)

# Version of this tool
__version__ = "21.11.0"


def nice_print_rot(value, threshold=1.0e-4):
    """
    Converts a float number to a LaTeX string, possibly converting "common" values (integers, and simple square roots)
    to nicer form.

    :param value: a float value
    :param threshold: a numerical threshold to decide if a number is an integer, a square root, ...
    :return: a (LaTeX) string
    """
    int_value = int(round(value))

    if abs(int_value - value) < threshold:
        return f"{int_value:d}"
    if abs(value - 0.5) < threshold:
        return r"\frac{1}{2}"
    if abs(value - (-0.5)) < threshold:
        return r"-\frac{1}{2}"
    if abs(value - math.sqrt(2) / 2) < threshold:
        return r"\frac{\sqrt{2}}{2}"
    if abs(value - (-math.sqrt(2) / 2)) < threshold:
        return r"-\frac{\sqrt{2}}{2}"
    if abs(value - math.sqrt(3) / 2) < threshold:
        return r"\frac{\sqrt{3}}{2}"
    if abs(value - (-math.sqrt(3) / 2)) < threshold:
        return r"-\frac{\sqrt{3}}{2}"

    # As a fallback, return the float representation
    return f"{value:10.5f}"


def process_structure_core(
    structure, logger, flask_request
):  # pylint: disable=unused-argument, too-many-locals, too-many-statements
    start_time = time.time()

    # Get information on the crystal structure to be shown later
    inputstructure_cell_vectors = [
        [idx, coords[0], coords[1], coords[2]]
        for idx, coords in enumerate(structure[0], start=1)
    ]
    inputstructure_symbols = [chemical_symbols[num] for num in structure[2]]
    inputstructure_atoms_scaled = [
        [label, coords[0], coords[1], coords[2]]
        for label, coords in zip(inputstructure_symbols, structure[1])
    ]

    inputstructure_positions_cartesian = np.dot(
        np.array(structure[1]), np.array(structure[0]),
    ).tolist()
    inputstructure_atoms_cartesian = [
        [label, coords[0], coords[1], coords[2]]
        for label, coords in zip(
            inputstructure_symbols, inputstructure_positions_cartesian
        )
    ]

    # prepare template dictionary to return later
    return_data = {
        "app_data_json": json.dumps(
            None
        ),  # None by default, if e.g. layers are not found
        "common_layers_search": None,  # None by default
        "layers": [],  # Empty list if no layers found
        "has_common_layers": False,
        "xsfstructure": get_xsf_structure(structure),
        "inputstructure_cell_vectors": inputstructure_cell_vectors,
        "inputstructure_atoms_scaled": inputstructure_atoms_scaled,
        "inputstructure_atoms_cartesian": inputstructure_atoms_cartesian,
        "ase_version": ase.__version__,
        "tools_barebone_version": get_tools_barebone_version(),
        "this_tool_version": __version__,
    }

    asecell = ase_from_tuple(structure)

    # Get the primitive cell from the ase cell obtained from the user
    # NOTE! Beside getting the primitive cell, this function will also refine its symmetry.
    primitive_tuple = spglib.find_primitive(
        (
            asecell.get_cell(),
            asecell.get_scaled_positions(),
            asecell.get_atomic_numbers(),
        ),
        symprec=SYMPREC,
    )
    # Get now the conventional cell (it re-does a symmetry analysis)
    dataset = spglib.get_symmetry_dataset(primitive_tuple)
    conventional_tuple = (
        dataset["std_lattice"],
        dataset["std_positions"],
        dataset["std_types"],
    )
    conventional_asecell = ase_from_tuple(conventional_tuple)

    bulk_spg = SpacegroupAnalyzer(
        AseAtomsAdaptor().get_structure(conventional_asecell), symprec=SYMPREC
    )
    pg_bulk_number = pg_number_from_hm_symbol(bulk_spg.get_point_group_symbol())
    return_data["pointgroup_bulk"] = prepare_pointgroup(pg_bulk_number)
    return_data["spacegroup_bulk"] = prepare_spacegroup(bulk_spg)

    # NOTE: there are cases in which it might not be detected - we'll deal with how to display those in the UI

    # From now on, I will work with the conventional cell rather than the one specified by the user
    # This is important because we sometimes (in the output) make assumptions that the number of layers found
    # is the number of layers in the conventional cell (e.g. when we say "Multilayer spacegroup
    # for N >= {num_layers_conventional}").
    is_layered, layer_structures, layer_indices, rotated_asecell = find_layers(
        conventional_asecell
    )

    detected_hall_number = None
    if rotated_asecell is not None:
        # Detect Hall setting
        for hall_number in hall_numbers_of_spacegroup[dataset["number"]]:
            hall_dataset = get_symmetry_dataset(
                tuple_from_ase(rotated_asecell), hall_number=hall_number
            )
            # print(hall_number, hall_dataset['transformation_matrix'], hall_dataset['origin_shift'])

            # If it's Identity, we've identified the correct Hall setting (or at least one among
            # the possible origin choices). We stop at the first one that satisfied this.
            if (
                np.sum(
                    (np.eye(3) - np.array(hall_dataset["transformation_matrix"])) ** 2
                )
                < 1.0e-6
            ):
                detected_hall_number = hall_number
                break

    return_data["hall_number"] = detected_hall_number

    # Get the scaled radii for the bonds detection
    scaled_radii_per_site = get_covalent_radii_array(asecell)
    # This is a dict of the form {"Na": 1.3, "C": 1.5}, ..
    scaled_radii_per_kind = {
        atom.symbol: scaled_radius
        for atom, scaled_radius in zip(asecell, scaled_radii_per_site)
    }

    # I now construct the list of *pairwise* threshold distances, to be passed to JSMOL
    # In theory, I could use simply "set bondTolerance 0;" and "{_P}.bondingRadius = 1.4158" (in this example, for
    # the P element). However, it does not seem to be setting the threshold for showing a bond at the sum,
    # but at some different value.
    # Therefore, I instead compute the pairwise threshold distance, say for elements Ga and As, and pass the following
    # JSMOL string (if, say, I don't want bonds for atoms closer than 0.2 ang, and the threshold distance is 2.27 ang):
    # "connect 0.2 2.27 {_Ga} {_As};"
    # It is good to prepend this with "set autobond off;" before loading, or use first a "connect delete;" to remove
    # existing bonds
    jsmol_bond_commands = []
    min_bonding_distance = 0.2  # ang
    for kind1, radius1 in scaled_radii_per_kind.items():
        for kind2, radius2 in scaled_radii_per_kind.items():
            if kind1 > kind2:
                # Just do one of the two pairs
                continue
            jsmol_bond_commands.append(
                f"connect {min_bonding_distance} {radius1+radius2} {{_{kind1}}} {{_{kind2}}}; "
            )

    # Encode as JSON string before sending, so it's safe to inject in the code
    return_data["jsmol_bond_command"] = json.dumps("".join(jsmol_bond_commands))

    if not is_layered:
        # I return here; some sections will not be present in the output so they will not be shown.
        compute_time = time.time() - start_time
        return_data["compute_time"] = compute_time
        logger.debug(json.dumps(return_data, indent=2, sort_keys=True))
        return return_data

    rot, transl, center, message = find_common_transformation(
        rotated_asecell, layer_indices
    )

    # Bring back atomic positions so that the origin is the center
    # of the coincidence operation (if found) and atomic positions
    # are inside the the unit cell in the layer plane
    if center is not None:
        rotated_asecell.positions -= center
        rotated_asecell.pbc = [True, True, False]
        rotated_asecell.positions = rotated_asecell.get_positions(wrap=True)
        rotated_asecell.pbc = [True, True, True]
        for layer in layer_structures:
            layer.positions -= center
            layer.pbc = [True, True, False]
            layer.positions = layer.get_positions(wrap=True)
            layer.pbc = [True, True, True]

    layer_xsfs = [
        get_xsf_structure(tuple_from_ase(layer_structure))
        for layer_structure in layer_structures
    ]

    return_data["layers"] = list(
        zip(
            layer_xsfs,
            # Needed because this might return int64 numpy objects, not JSON-serializable
            [
                [int(index) for index in this_layer_indices]
                for this_layer_indices in layer_indices
            ],
        )
    )

    # This is returned both in the return_data, for the HTML view, and in the app data,
    # to be set as a minimum for the REST API requests
    num_layers_bulk = len(layer_indices)
    return_data["num_layers_bulk"] = num_layers_bulk

    return_data["rotated_cell"] = {
        "layer_cell": rotated_asecell.cell.tolist(),
        "layer_atoms": [
            list(
                zip(
                    rotated_asecell[this_layer_indices].symbols,
                    rotated_asecell[this_layer_indices].positions.tolist(),
                )
            )
            for this_layer_indices in layer_indices
        ],
    }

    # I return here; some sections will not be present in the output so they will not be shown.
    compute_time = time.time() - start_time
    return_data["compute_time"] = compute_time
    logger.debug(json.dumps(return_data, indent=2, sort_keys=True))
    return return_data


