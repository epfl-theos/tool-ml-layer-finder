import json
import math
import time

import ase
import ase.io
import numpy as np
import shap
import spglib


from ase import Atoms
from ase.data import chemical_symbols

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .utils.structures import (
    ase_from_tuple,
    get_xsf_structure,
    tuple_from_ase,
)
from tools_barebone import get_tools_barebone_version
from .utils.lowdimfinder import (
    LowDimFinder,
    _map_atomic_number_radii_van_der_Waals_alvarez,
)
from .utils.pointgroup import (
    pg_number_from_hm_symbol,
    prepare_pointgroup,
    prepare_spacegroup,
    SYMPREC,
)

###Featurization
import matminer
from matminer.featurizers.base import MultipleFeaturizer

from matminer.featurizers.structure import (
    SiteStatsFingerprint,
    ChemicalOrdering,
    MaximumPackingEfficiency,
)

### ML Model
import joblib

# Version of this tool
__version__ = "21.11.0"


def get_feature_name(pos):  # pylint: disable=too-many-branches
    """Return the name of the feature, given its index `pos`."""

    local_env_prop_names = [
        "Atomic number",
        "Mendeleev number",
        "Atomic weight",
        "Melting temperature",
        "Periodic table column",
        "Periodic table row",
        "Covalent radius",
        "Electronegativity",
        "Number of filled s valence orbitals",
        "Number of filled p valence orbitals",
        "Number of filled d valence orbitals",
        "Number of filled f valence orbitals",
        "Number of valence electrons",
        "Number of unfilled s valence orbitals",
        "Number of unfilled p valence orbitals",
        "Number of unfilled d valence orbitals",
        "Number of unfilled f valence orbitals",
        "Number of unfilled valence orbitals",
        "DFT volume per atom",
        "DFT band gap",
        "DFT magnetic moment",
        "Spacegroup number",
    ]
    assert len(local_env_prop_names) == 22
    local_env_stats = ["min", "max", "range", "mean", "mean abs deviation"]
    assert len(local_env_stats) == 5

    if pos == 0:
        return "Chemical ordering (first neighbors)"
    if pos == 1:
        return "Chemical ordering (second neighbors)"
    if pos == 2:
        return "Chemical ordering (third neighbors)"
    if pos == 3:
        return "Max packing efficiency"
    # 4 above, 22*5 = 110 more, if the index is > 114 it's out of bounds
    if pos >= 114:
        raise ValueError("Only 114 features known")

    idx = pos - 4
    prop_idx = idx % len(local_env_prop_names)
    stats_idx = idx // len(local_env_prop_names)
    return f"{local_env_prop_names[prop_idx]} ({local_env_stats[stats_idx]})"


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
):  # pylint: disable=unused-argument, too-many-locals, too-many-statements, too-many-branches
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
        np.array(structure[1]),
        np.array(structure[0]),
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
        "matminer_version": matminer.__version__,
        "joblib_version": joblib.__version__,
        "tools_barebone_version": get_tools_barebone_version(),
        "this_tool_version": __version__,
        "ML_predictions": False,
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

    ### MOHAMMAD: Run LowDimFinder
    for radiiOffset in [-0.75, -0.7, -0.65, -0.6, -0.55]:

        low_dim_finder = LowDimFinder(
            aiida_structure=conventional_asecell,
            vacuum_space=40.0,
            radii_offset=radiiOffset,
            bond_margin=0.0,
            max_supercell=3,
            min_supercell=3,
            rotation=True,
            full_periodicity=False,
            radii_source="alvarez",
            orthogonal_axis_2D=True,
        )

        ### MOHAMMAD: Replace four variables (is_layered, layer_structures, layer_indices, rotated_asecell) with LowDimFinder Results!

        low_dim_finder_results = low_dim_finder.get_group_data()

        if 2 in low_dim_finder_results["dimensionality"]:
            is_layered = True

            layer_structures = []
            layer_indices = []
            for i in range(len(low_dim_finder_results["dimensionality"])):
                if low_dim_finder_results["dimensionality"][i] == 2:
                    struc = Atoms(
                        symbols=low_dim_finder_results["chemical_symbols"][i],
                        positions=low_dim_finder_results["positions"][i],
                        cell=low_dim_finder_results["cell"][i],
                        tags=low_dim_finder_results["tags"][i],
                    )
                    layer_structures.append(struc)
                    layer_indices.append(
                        low_dim_finder._get_unit_cell_groups()[  # pylint: disable=protected-access
                            i
                        ]
                    )
                    rotated_asecell = low_dim_finder._rotated_structures[  # pylint: disable=protected-access
                        i
                    ]
            break
        if radiiOffset == -0.55:
            is_layered = False
            layer_indices = None
            layer_structures = None
            rotated_asecell = None

    ### MOHAMMAD: Just to be consistant with before and avoid further changes!
    ### MOHAMMAD: layer_indices must be smaller than the number of elements in a unitcell!
    ### MOHAMMAD: For example, change: layer_indices of [[0, 4, 8, 11, 15, 19, 20, 75, 79, 84, 88, 95],
    ### MOHAMMAD: [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22]] to [[0, 4, 8, 11, 15, 19, 20, 3, 7, 12, 16, 23],
    ### MOHAMMAD: [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22]] -- In this case the number of element is 24!

    ### MOHAMMAD: More efficient way:

    if is_layered:
        for i in range(len(layer_indices)):  # pylint: disable=consider-using-enumerate
            for j in range(len(layer_indices[i])):
                if layer_indices[i][j] >= len(conventional_asecell):
                    tmp = layer_indices[i][j]
                    layer_indices[i][j] = tmp % len(conventional_asecell)

    ### MOHAMMAD: replace all the components and commented!

    # Get the scaled radii for the bonds detection

    ### MOHAMMAD: Used vdW radii in order to draw bonds by visualizer

    scaled_radii_per_site = np.array(
        [
            _map_atomic_number_radii_van_der_Waals_alvarez.get(atom.number)
            for atom in asecell
        ]
    )

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

    ### MOHAMMAD: Add offset to the vdw radii

    jsmol_bond_commands = []
    min_bonding_distance = 0.2  # ang
    for kind1, radius1 in scaled_radii_per_kind.items():
        for kind2, radius2 in scaled_radii_per_kind.items():
            if kind1 > kind2:
                # Just do one of the two pairs
                continue
            jsmol_bond_commands.append(
                f"connect {min_bonding_distance} {radius1+radius2+2*radiiOffset} {{_{kind1}}} {{_{kind2}}}; "
            )

    # Encode as JSON string before sending, so it's safe to inject in the code
    return_data["jsmol_bond_command"] = json.dumps("".join(jsmol_bond_commands))

    if not is_layered:
        # I return here; some sections will not be present in the output so they will not be shown.
        compute_time = time.time() - start_time
        return_data["compute_time"] = compute_time
        logger.debug(json.dumps(return_data, indent=2, sort_keys=True))
        return return_data

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

    ### MOHAMMAD: Here we generate features for the strucutres that passed the LowDimFinder

    featurizer = MultipleFeaturizer(
        [
            ChemicalOrdering(),
            MaximumPackingEfficiency(),
            SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
        ]
    )

    structures_pg = AseAtomsAdaptor.get_structure(conventional_asecell)

    list_structures = {}
    list_structures["structure"] = structures_pg

    print(structures_pg)
    X = featurizer.featurize_many([structures_pg], ignore_errors=True)
    print(X)

    ### MOHAMMAD: Load the trained model

    loaded_RF = joblib.load(
        "/home/app/code/webservice/static/random_forest_model.joblib"
    )
    explainer = shap.Explainer(loaded_RF)

    ### MOHAMMAD: make prediction!
    pred_RF = loaded_RF.predict(X)
    if pred_RF == [1]:
        return_data["ML_predictions"] = True
    else:
        return_data["ML_predictions"] = False

    # Also get the SHAP values
    shap_explanation = explainer(np.array(X))
    # This is now an array of the |shap|
    abs_shap_values = np.abs(shap_explanation[0, :, 1].values)

    MAX_DISPLAY = 20
    sorted_shaps = sorted(list(zip(abs_shap_values, range(len(abs_shap_values)))))[::-1]

    sorted_shap_with_feature_name = []
    for idx in range(MAX_DISPLAY):
        abs_shap, feature_pos = sorted_shaps[idx]
        sorted_shap_with_feature_name.append([get_feature_name(feature_pos), abs_shap])
    sorted_shap_with_feature_name.append(
        [
            f"Sum of {len(sorted_shaps) - MAX_DISPLAY} other features",
            sum(shap[0] for shap in sorted_shaps[MAX_DISPLAY:]),
        ]
    )
    return_data["sorted_abs_shaps"] = sorted_shap_with_feature_name

    # I return here; some sections will not be present in the output so they will not be shown.
    compute_time = time.time() - start_time
    return_data["compute_time"] = compute_time
    logger.debug(json.dumps(return_data, indent=2, sort_keys=True))
    return return_data
