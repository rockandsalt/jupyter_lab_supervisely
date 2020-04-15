FROM supervisely/jupyterlab:latest

RUN pip install --upgrade numpy scikit-image numba porespy mpld3
