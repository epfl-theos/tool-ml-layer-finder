# ML-layer-finder online tool
A tool to find layers (and low-dimensional structures) within a bulk 3D structure, combined with machine learning to predict if the layer has a low binding energy

[![Actions Status](https://github.com/epfl-theos/tool-ml-layer-finder/workflows/Continuous%20integration/badge.svg)](https://github.com/epfl-theos/tool-ml-layer-finder/actions)

## About the tool

This tool allows users to upload the bulk crystal structure in several standard formats (or to choose from a few examples), and then layered structures are identified based on geometrical criteria. Finally, after generating features vectors representing the crystal structure, the tool uses a machine-learning model to see if the crystal structure can be exfoliated or have high binding energy.

The output page includes relevant information on the structure (interactive visualizations of the bulk multilayer) and whether the structure is suitable for exfoliation or not based on the geometrical criteria. If yes, the corresponding two-dimensional layers are displayed. In addition, the machine-learning model is run to predict if the structure might actually have a low binding energy, and results are displayed.

## Online version
This tool is deployed on the Materials Cloud "Tools" section [here](https://ml-layer-finder.materialscloud.io/), so you can use it without need of installation.

## How to cite
If you use this tool, please cite the following work:

* **M. T. Vahdat, K. A. Varoon, and G. Pizzi, *Machine-learning accelerated identification of exfoliable two-dimensional materials*, submitted (2022).**

You might also want to cite the [ASE](https://wiki.fysik.dtu.dk/ase/), [pymatgen](http://pymatgen.org), [matminer](https://github.com/hackingmaterials/matminer) and [shap](https://shap.readthedocs.io/) libraries that are used internally by the tool, as well as <a href="https://doi.org/10.1038/s41565-017-0035-5" target="_blank">N. Mounet <em>et al.</em>, <em>Two-dimensional materials from high-throughput computational exfoliation of experimentally known compounds</em>, Nature Nanotech. 13, 246-252 (2018)</a> where the geometrical-screening code was originally developed, and from which the DFT data for the binding energies was extracted to train our model.

## How to deploy on your computer
1. Install [Docker](https://www.docker.com)
2. Clone this repository
3. Run `./admin-tools/build-and-run.sh`. This will build the Docker image, start the docker container, and open a browser to the correct URL.
   If the browser does not open automatically, connect to http://localhost:8098

If you want to check the Apache logs, you can run `./admin-tools/get-apache-logs.sh --reload` to see the logs of the tool in real time while you use it.

## Acknowledgements
This tool uses the ASE and pymatgen libraries for structure manipulation, and matminer for featurization.
The tool is based upon the [tools-barebone framework](https://github.com/materialscloud-org/tools-barebone) developed by the Materials Cloud team.

We acknowledge funding from the [MARVEL National Centre of Competence in Research](https://nccr-marvel.ch) of the Swiss National Science Foundation (SNSF), the [European Centre of Excellence MaX "Materials design at the Exascale"](http://www.max-centre.eu), the [swissuniversities P-5 "Materials Cloud" project](https://www.materialscloud.org/swissuniversities).
