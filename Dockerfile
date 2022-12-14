FROM python:3.8

# install other python packages
ADD requirements.txt .
RUN pip install --no-cache -r requirements.txt

RUN pip install jupyter jupyter_contrib_nbextensions "nbconvert<6" && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable collapsible_headings/main