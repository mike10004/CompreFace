ARG BASE_IMAGE
FROM ${BASE_IMAGE:-python:3.8-slim}

RUN apt-get update && apt-get install -y build-essential cmake git wget unzip \
        curl yasm pkg-config libswscale-dev libtbb2 libtbb-dev libjpeg-dev \
        libpng-dev libtiff-dev libavformat-dev libpq-dev libfreeimage3 \
    && rm -rf /var/lib/apt/lists/*

# install common python packages
SHELL ["/bin/bash", "-c"]
WORKDIR /app/ml
COPY requirements-console.txt .
RUN pip --no-cache-dir install -r requirements-console.txt

ARG BE_VERSION
ARG APP_VERSION_STRING
ENV BE_VERSION=$BE_VERSION
ENV APP_VERSION_STRING=$APP_VERSION_STRING
ENV HOME=/app/ml
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=0
ENV JOBLIB_MULTIPROCESSING=0

# download ML models
ARG INTEL_OPTIMIZATION=false
ARG GPU_IDX=-1
ENV GPU_IDX=$GPU_IDX INTEL_OPTIMIZATION=$INTEL_OPTIMIZATION
ARG FACE_DETECTION_PLUGIN="facenet.FaceDetector"
ARG CALCULATION_PLUGIN="facenet.Calculator"
ARG EXTRA_PLUGINS="facenet.LandmarksDetector,agegender.AgeDetector,agegender.GenderDetector,facenet.facemask.MaskDetector"
ENV FACE_DETECTION_PLUGIN=$FACE_DETECTION_PLUGIN CALCULATION_PLUGIN=$CALCULATION_PLUGIN \
    EXTRA_PLUGINS=$EXTRA_PLUGINS
ENV COMPREFACE_HABITAT=console
COPY src src
COPY srcext srcext
COPY werkzeug_stub werkzeug_stub
RUN python -m src.services.facescan.plugins.setup

# copy rest of the code
COPY tools tools
COPY sample_images sample_images

# run tests
ARG SKIP_TESTS
COPY pytest.ini .
RUN if [ -z $SKIP_TESTS  ]; then pip --no-cache-dir install -r requirements-dev.txt; pytest -m "not performance" /app/ml/src; fi

# default command: print help
CMD ["python", "-m", "tools.scan"]
