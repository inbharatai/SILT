# Isolated GLM-5.3-Flash worker image.
#
# This runtime is DELIBERATELY separate from the main SILT environment:
# GLM-5.3-Flash needs Transformers 5.x (pinned to the model card's
# transformers_version, 5.16.0) while the main environment pins
# transformers==4.51.3. The two must never share an interpreter.
#
# Build:  docker build -f workers/glm53/Dockerfile -t silt-glm53-worker .
# Run:    docker run --rm -i \
#           -v /path/to/GLM-5.3-Flash-BF16:/checkpoint:ro \
#           -e GLM_CHECKPOINT=/checkpoint \
#           silt-glm53-worker < requests.jsonl > responses.jsonl
#
# The checkpoint is mounted READ-ONLY (:ro): the teacher's weights are
# read-only by construction, not just by convention.

FROM python:3.12-slim

# The worker runs fully offline: the checkpoint is mounted, never fetched.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

COPY workers/glm53/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

# The worker needs only asea's stdlib/pydantic core (worker_protocol,
# adapters, intervention, errors) -- never the localmodels extras.
# build_support.py (the custom build command declared in pyproject.toml
# under [tool.setuptools] build_support) is a single MODULE at the repo
# root, not a directory: ``pip install /silt`` fails without it.
COPY pyproject.toml README.md /silt/
COPY src/ /silt/src/
COPY build_support.py /silt/build_support.py
RUN pip install --no-cache-dir /silt

COPY workers/glm53/worker.py /worker/worker.py

# Non-root; the checkpoint tree is :ro anyway.
RUN useradd --create-home --uid 1000 worker
USER worker

ENTRYPOINT ["python", "/worker/worker.py"]