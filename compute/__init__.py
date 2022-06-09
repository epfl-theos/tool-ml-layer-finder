import logging
import os
import traceback

import flask

from .ml_layer_finder_engine import process_structure_core
from .utils.structures import parse_structure
from .utils.response import FlaskRedirectException

# Ignoring the import error because it's from tools-barebone
# Not ideal, but works for now
from web_module import get_config  # pylint: disable=import-error
import header

VALID_EXAMPLES = {
    "WTe2": "WTe2-02f1827d-f339-436f-baf6-66d1cf142fcf_structure.xsf",
    "ZnCl2": "ZnCl2-e5f429a4-3b02-4fb0-8921-0a7ab05078ed_structure.xsf",
    # MoS2 bulk from Materials Cloud:
    # https://www.materialscloud.org/explore/2dstructures/details/6e58409f-4ab2-4883-9686-87d4d89c0bf9
    # (Originally from COD, 9007660, P6_3/mmc)
    "MoS2": "MoS2-6e58409f-4ab2-4883-9686-87d4d89c0bf9_structure.xsf",
    # black P bulk from Materials Cloud:
    # https://www.materialscloud.org/explore/2dstructures/details/904c1f0e-da23-42f0-95b4-a4fee98e6d04
    # (Originally from COD, 9012486, Cmce)
    "blackP": "P-904c1f0e-da23-42f0-95b4-a4fee98e6d04_structure.xsf",
    "graphite": "graphite-544d62e4-8ebe-404c-aa17-b99be62ea70b.xsf",
    "BN": "BN-P6_3mmc-f7e2ff32-27ed-4c89-9c3c-4acbaffbb897.xsf",
    "Sr2Nb5O9": "Sr2Nb5O9-866e918e-7a5f-41e3-980e-038852391b5a_structure.xsf",
    "KGaSe2": "KGaSe2-1e35f667-92e1-4b90-bfc8-daf3c5d1f7b0_structure.xsf",
    "Na2TiS2O": "Na2TiS2O-8d5eb648-9aa3-409f-9e72-861e38e11f30_structure.xsf",
    "B2N2": "B2N2-8f2e38e9-01d5-4208-adaf-daa461ac8139_structure.xsf",
    "AgBiTe3": "AgBiTe3-0bddefb1-3b12-4d9b-bd77-2491d5d9fdb9_structure.xsf",
}

logger = logging.getLogger("tool-ml-layer-finder-tool-app")
blueprint = flask.Blueprint("compute", __name__, url_prefix="/compute")


@blueprint.route("/process_structure/", methods=["GET", "POST"])
def process_structure():
    if flask.request.method == "POST":
        # check if the post request has the file part
        if "structurefile" not in flask.request.files:
            return flask.redirect(flask.url_for("input_data"))
        structurefile = flask.request.files["structurefile"]
        fileformat = flask.request.form.get("fileformat", "unknown")
        filecontent = structurefile.read().decode("utf-8")

        try:
            structure = parse_structure(
                filecontent=filecontent,
                fileformat=fileformat,
                extra_data=dict(flask.request.form),
            )
        except Exception as exc:
            traceback.print_exc()
            flask.flash(
                "Unable to parse the structure, sorry... ({}, {})".format(
                    str(type(exc)), str(exc)
                )
            )
            return flask.redirect(flask.url_for("input_data"))

        try:
            data_for_template = process_structure_core(
                structure=structure,
                logger=logger,
                flask_request=flask.request,
            )
            config = get_config()
            tvars = header.template_vars
            return flask.render_template(
                "user_templates/visualizer_header.j2",
                **data_for_template,
                **config,
                **tvars,
            )
        except FlaskRedirectException as e:
            flask.flash(str(e))
            return flask.redirect(flask.url_for("input_data"))
        except Exception as exc:
            traceback.print_exc()
            flask.flash(
                "Unable to process the structure, sorry... ({}, {})".format(
                    str(type(exc)), str(exc)
                )
            )
            return flask.redirect(flask.url_for("input_data"))
    else:  # GET Request
        return flask.redirect(flask.url_for("input_data"))


@blueprint.route("/process_example_structure/", methods=["GET", "POST"])
def process_example_structure():
    """
    Process an example structure (example name from POST request)
    """
    if flask.request.method == "POST":
        examplestructure = flask.request.form.get("examplestructure", "<none>")
        fileformat = "xsf-ase"

        try:
            filename = VALID_EXAMPLES[examplestructure]
        except KeyError:
            flask.flash("Invalid example structure '{}'".format(examplestructure))
            return flask.redirect(flask.url_for("input_data"))

        # I expect that the valid_examples dictionary already filters only
        # existing files, so I don't try/except here
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "xsf-examples",
                filename,
            )
        ) as structurefile:
            filecontent = structurefile.read()

        try:
            structure = parse_structure(
                filecontent=filecontent,
                fileformat=fileformat,
            )
        except Exception as exc:
            flask.flash(
                "Unable to parse the example structure, sorry... ({}, {})".format(
                    str(type(exc)), str(exc)
                )
            )
            return flask.redirect(flask.url_for("input_data"))

        try:
            data_for_template = process_structure_core(
                structure=structure, logger=logger, flask_request=flask.request
            )
            config = get_config()
            tvars = header.template_vars
            return flask.render_template(
                "user_templates/visualizer_header.j2",
                **data_for_template,
                **config,
                **tvars,
            )
        except FlaskRedirectException as e:
            flask.flash(str(e))
            return flask.redirect(flask.url_for("input_data"))
        except Exception as exc:
            traceback.print_exc()
            flask.flash(
                "Unable to process the structure, sorry... ({}, {})".format(
                    str(type(exc)), str(exc)
                )
            )
            return flask.redirect(flask.url_for("input_data"))
    else:  # GET Request
        return flask.redirect(flask.url_for("input_data"))
