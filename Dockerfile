FROM materialscloud/tools-barebone:1.3.0

LABEL maintainer="Giovanni Pizzi <giovanni.pizzi@epfl.ch>"

COPY ./tool-requirements.txt /home/app/code/webservice/static/tool-requirements.txt
RUN pip3 install -r /home/app/code/webservice/static/tool-requirements.txt

# Trained Model
COPY ./model/random_forest_model.joblib /home/app/code/webservice/static/random_forest_model.joblib

# Copy various files: configuration, user templates, the actual python code, ...

COPY ./config.yaml /home/app/code/webservice/static/config.yaml
COPY ./user_templates/ /home/app/code/webservice/templates/user_templates/
COPY ./compute/ /home/app/code/webservice/compute/
COPY ./user_static/ /home/app/code/webservice/user_static/

RUN cp /home/app/code/webservice/templates/header.html /home/app/code/webservice/templates/header_pages.html && \
    sed -i "s|base.html|user_templates/base.html|g" /home/app/code/webservice/templates/header_pages.html && \
    sed -i "s|static/|../../static/|g" /home/app/code/webservice/templates/header_pages.html


# Set proper permissions on files just copied
RUN chmod -R o+rX /home/app/code/webservice/
RUN chown -R app:app /home/app/code/webservice/
