#!/usr/bin/env bash
# Boot a single-node HydraDB on local disk, wait for Bolt, then serve Loadout.
set -euo pipefail
DATA=${HYDRADB_DATA:-/tmp/hydradb}
mkdir -p "$DATA/store" "$DATA/cache"
TOKEN=${HYDRADB_TOKEN:-$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')}
printf '%s\n' "$TOKEN" > "$DATA/auth-token"

export CLOUD_PROVIDER=local LOCAL_PATH="$DATA/store" \
  GRAPH_NAMESPACE=default GRAPH_ID=default GRAPH_CELL_ID=cell-0 GRAPH_CELLS=cell-0 GRAPH_NODE_ID=node-0 \
  GRAPH_BOLT_NODE_ADDRESSES=node-0=127.0.0.1:7687 GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:7687 \
  GRAPH_DATA_CACHE_DIR="$DATA/cache" GRAPH_AUTH_TOKEN_FILE="$DATA/auth-token" GRAPH_ALLOW_PLAINTEXT=true

/usr/local/bin/graph-node > "$DATA/graph-node.log" 2>&1 &
NODE=$!
for i in $(seq 1 120); do
  if python3 -c "import socket;socket.create_connection(('127.0.0.1',7687),1)" 2>/dev/null; then break; fi
  if ! kill -0 "$NODE" 2>/dev/null; then echo "graph-node exited"; tail -50 "$DATA/graph-node.log"; exit 1; fi
  sleep 1
done
echo "HydraDB is up"

export LOADOUT_HYDRADB_URI=bolt://127.0.0.1:7687 LOADOUT_HYDRADB_TOKEN="$TOKEN"
if [ "${LOADOUT_EVAL_ON_BOOT:-1}" = "1" ]; then
  (python3 -m loadout.evaluate "${LOADOUT_EVAL_TASKS:-600}" > "$DATA/eval.log" 2>&1 && echo "eval done") &
fi
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
