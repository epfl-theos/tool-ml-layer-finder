# ML-layer-finder online tool
A tool to find layers (and low-dimensional structures) within a bulk 3D structure, combined with machine learning to predict if the layer has a low binding energy

[![Actions Status](https://github.com/epfl-theos/tool-ml-layer-finder/workflows/Continuous%20integration/badge.svg)](https://github.com/epfl-theos/tool-ml-layer-finder/actions)

## About the tool

This tool allows users to upload the bulk crystal structure in several standard formats (or to choose from a few examples), and then layered structures are identified based on geometrical criteria. Finally, after generating features vectors representing the crystal structure, the tool uses a machine learning model to see if the crystal structure can be exfoliated or have high binding energy.

The demonstrated outcome page includes relevant information on the structure (interactive visualizations of the bulk multilayer) and whether the structure is suitable for exfoliation or not. If yes, the corresponding two-dimensional materials are displayed.

## Online version
This tool is deployed on the Materials Cloud "Tools" section [here](https://ml-layer-finder.materialscloud.io/), so you can use it without need of installation.

## How to cite
If you use this tool, please cite the following work:

* ...

You might also want to cite the ASE, pymatgen and matminer libraries that are used internally by the tool.

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
