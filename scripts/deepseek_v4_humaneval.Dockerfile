# Official human-eval requirements, version-pinned for this evaluation.
# Build without model data or credentials in the build context.
FROM python@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534
RUN python -m pip install --no-cache-dir numpy==2.2.6 fire==0.7.0 tqdm==4.67.1 termcolor==3.1.0
