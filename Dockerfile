FROM ubuntu:20.04

# install requirements
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
    build-essential \
    git \
    intel-mkl \
    libomp-dev \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-setuptools \
    swig \
    wget

RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3-linux-x86_64.tar.gz -O - | tar xzf - --strip=1 -C /usr
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0

# install FAISS (patched version)
RUN git clone --branch v1.7.1 https://github.com/facebookresearch/faiss.git /opt/faiss
WORKDIR /opt/faiss
ADD remove_nbits_assert.patch /opt/faiss
RUN git apply remove_nbits_assert.patch

RUN cmake \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_OPT_LEVEL=avx2 \
    -B build . && \
    make -C build -j faiss && \
    make -C build -j swigfaiss && \
    cd build/faiss/python && \
    python setup.py install
RUN make -C build install
ENV PYTHONPATH=/opt/faiss/build/faiss/python:$PYTHONPATH

# install other python packages
ADD requirements.txt .
RUN pip install --no-cache -r requirements.txt

RUN pip install jupyter jupyter_contrib_nbextensions "nbconvert<6" && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable collapsible_headings/main