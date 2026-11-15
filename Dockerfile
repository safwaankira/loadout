# The open-source HydraDB engine, with Loadout running beside it in the same container.
FROM ghcr.io/hydra-db/hydradb:0.4.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chown -R graph:graph /app
USER graph

ENV PATH=/venv/bin:$PATH \
    PORT=10000 \
    RUST_MIN_STACK=33554432
ENTRYPOINT []
CMD ["bash", "/app/start.sh"]
