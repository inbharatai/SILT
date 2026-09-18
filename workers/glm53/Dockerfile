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
# root, not a directory: ``pip install /silt`` fails without it. Its
# build hook additionally snapshots every file in
# src/asea/_package_resources.py AUDIT_ORIGINS into the wheel and FAILS
# if any of them is missing from the build context -- so the image must
# copy those exact inputs too (docs/SPECIALIST_WORKFLOW.md, the
# specialist experiment script, and the four audit test files), or the
# wheel build inside the image dies with "required source audit input
# missing".
COPY pyproject.toml README.md build_support.py /silt/
COPY src/ /silt/src/
COPY docs/SPECIALIST_WORKFLOW.md /silt/docs/SPECIALIST_WORKFLOW.md
COPY scripts/run_specialist_quality_experiment.py /silt/scripts/run_specialist_quality_experiment.py
COPY tests/test_specialist_workflow.py /silt/tests/test_specialist_workflow.py
COPY tests/test_specialist_quality_experiment_runner.py /silt/tests/test_specialist_quality_experiment_runner.py
COPY tests/test_hardware_dispatch.py /silt/tests/test_hardware_dispatch.py
COPY tests/test_controller_integrity_review.py /silt/tests/test_controller_integrity_review.py
RUN pip install --no-cache-dir /silt

COPY workers/glm53/worker.py /worker/worker.py

# Non-root; the checkpoint tree is :ro anyway.
RUN useradd --create-home --uid 1000 worker
USER worker

ENTRYPOINT ["python", "/worker/worker.py"]