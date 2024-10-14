FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL description="Docker container for MASt3R with dependencies installed. CUDA VERSION"
ENV DEVICE="cuda"
ENV MODEL="MASt3R_ViTLarge_BaseDecoder_512_dpt.pth"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git=1:2.34.1-1ubuntu1.10 \
    libglib2.0-0=2.72.4-0ubuntu2.2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch mast3r_sfm --recursive https://github.com/naver/mast3r /mast3r
WORKDIR /mast3r/dust3r
RUN pip install -r requirements.txt
RUN pip install -r requirements_optional.txt
RUN pip install opencv-python==4.8.0.74

WORKDIR /mast3r/dust3r/croco/models/curope/
RUN python setup.py build_ext --inplace

WORKDIR /mast3r
RUN pip install -r requirements.txt

# Prepare and empty machine for building.
RUN apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential \
      libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev \
      libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev \
      libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev; \
    rm -r /var/lib/apt/lists/*

# Bleeding edge version with new pycolmap bindings:
ARG COLMAP_GIT_COMMIT=3.10
ARG CMAKE_CUDA_ARCHITECTURES=89-real
ARG NINJA_BUILD_CONCURRENCY=
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git /colmap; \
    cd /colmap; \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT}; \
    git checkout FETCH_HEAD; \
    mkdir build; \
    cd build; \
    export CUDA_HOME="/usr/local/cuda"; \
    cmake .. -GNinja \
      -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -D CUDA_ENABLED=ON \
      -D CMAKE_BUILD_TYPE=Release; \
    ninja ${NINJA_BUILD_CONCURRENCY:+-j ${NINJA_BUILD_CONCURRENCY}}; \
    ninja install; \
    if [ -z "${PYCOLMAP_GIT_COMMIT:+set}" ]; then \
      # Newer embedded pycolmap
      cd /colmap/pycolmap; \
    else \
      git clone https://github.com/colmap/pycolmap.git /pycolmap; \
      cd /pycolmap; \
      git fetch https://github.com/colmap/pycolmap.git ${PYCOLMAP_GIT_COMMIT}; \
      git checkout FETCH_HEAD; \
      git submodule update --recursive --init; \
    fi; \
    pip wheel -w /wheelhouse .; \
    pycolmap_wheel=(/wheelhouse/pycolmap*.whl)

ARG GLOMAP_GIT_COMMIT=55e40b11a68b8819e71fdb30773258e20a284451
# Build and install GLOMAP.
RUN git clone https://github.com/colmap/glomap.git /glomap; \
    cd /glomap; \
    git fetch https://github.com/colmap/glomap.git ${GLOMAP_GIT_COMMIT}; \
    git checkout FETCH_HEAD; \
    mkdir build; \
    cd build; \
    export CUDA_HOME="/usr/local/cuda"; \
    cmake .. -GNinja \
      -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -D CUDA_ENABLED=ON \
      -D CMAKE_BUILD_TYPE=Release; \
    ninja ${NINJA_BUILD_CONCURRENCY:+-j ${NINJA_BUILD_CONCURRENCY}}; \
    ninja install

RUN pip install faiss-gpu cython
SHELL ["/usr/bin/env", "bash", "-xveuc"]

RUN git clone https://github.com/jenicek/asmk.git /asmk; \
    cd /asmk/cython/; \
    cythonize *.pyx; \
    cd ..; \
    pip install .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH}

RUN sed -i 's|demo.py|demo_glomap.py|' /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
