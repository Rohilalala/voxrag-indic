#!/usr/bin/env bash
# Provision the two Railway services voxrag needs. Idempotent: safe to re-run.
#
# It creates (or leaves alone, if they already exist):
#   * a Qdrant service from the stock image, with a volume and the storage settings that
#     keep the HNSW graph in RAM;
#   * this repo's API service, with a volume for the index and every environment variable
#     it needs, including the private URL of the Qdrant service.
#
# It does NOT deploy. Deploying streams build logs and is better watched by hand, so the
# script prints the exact `railway up` to run when it finishes.
#
# Secrets: the Sarvam key is passed to the CLI on stdin, never in argv and never echoed.
# `set -x` is deliberately not used anywhere in this file for the same reason.
#
# Overridable by environment:
#   VOXRAG_RAILWAY_PROJECT   project name to create if the directory is not linked (voxrag)
#   VOXRAG_API_SERVICE       API service name (voxrag-api)
#   VOXRAG_QDRANT_SERVICE    Qdrant service name (qdrant)
#   VOXRAG_QDRANT_IMAGE      Qdrant image (qdrant/qdrant:v1.12.4)
#   VOXRAG_SARVAM_API_KEY    STT key; falls back to the value in ./.env

set -euo pipefail

PROJECT_NAME="${VOXRAG_RAILWAY_PROJECT:-voxrag}"
API_SERVICE="${VOXRAG_API_SERVICE:-voxrag-api}"
QDRANT_SERVICE="${VOXRAG_QDRANT_SERVICE:-qdrant}"
# Pinned to the version docker-compose.yml runs, so the deployed server is the one every
# retrieval and latency number in the README was measured against.
QDRANT_IMAGE="${VOXRAG_QDRANT_IMAGE:-qdrant/qdrant:v1.12.4}"

#: Must match the Dockerfile's VOLUME and the entrypoint's VOXRAG_DATA_DIR default.
API_MOUNT="/app/data"
QDRANT_MOUNT="/qdrant/storage"
#: Qdrant's REST port, and the port Railway's edge proxy should target on the API.
QDRANT_PORT="6333"
API_PORT="8000"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m  x %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------------------
# Preconditions
# --------------------------------------------------------------------------------------

command -v railway >/dev/null 2>&1 || die "the railway CLI is not on PATH.
      Install it with one of:
        brew install railway
        npm i -g @railway/cli
        bash <(curl -fsSL https://railway.com/install.sh)
      then re-run this script."

# Used only to read the CLI's --json output. Bare grep on JSON breaks on whitespace
# changes, and a wrong answer here means creating a duplicate service.
command -v python3 >/dev/null 2>&1 || die "python3 is required to parse the CLI's JSON output."

say "checking that the railway CLI is logged in"
if ! railway whoami >/dev/null 2>&1; then
    die "not logged in.
      Run:  railway login
      (that opens a browser; on a machine with no browser use \`railway login --browserless\`,
      which prints a link and a short code that expires after ten minutes).
      Then re-run this script."
fi
info "logged in as: $(railway whoami 2>/dev/null | tail -n1)"

# --------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------

#: Every string value of key `name` anywhere in the JSON on stdin, one per line. Tolerant
#: on purpose: the CLI has moved these payloads between a bare list and a wrapper object,
#: and this only ever has to answer "does this name already exist".
json_names() {
    python3 -c '
import json, sys

def walk(node):
    if isinstance(node, dict):
        value = node.get("name")
        if isinstance(value, str):
            yield value
        for child in node.values():
            yield from walk(child)
    elif isinstance(node, list):
        for child in node:
            yield from walk(child)

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for name in walk(data):
    print(name)
'
}

#: Both of these abort rather than answer "no" when the CLI call itself fails: a false
#: negative here creates a duplicate service or a second volume, which is worse than stopping.
service_exists() {
    local listing
    listing="$(railway service list --json 2>/dev/null)" \
        || die "\`railway service list --json\` failed; refusing to guess whether \"$1\" exists."
    printf '%s' "$listing" | json_names | grep -Fxq "$1"
}

#: Volumes are matched on mount path, which is what has to be unique per service.
volume_mounted() {
    local listing
    listing="$(railway volume list --json 2>/dev/null)" \
        || die "\`railway volume list --json\` failed; refusing to guess whether $1 is mounted."
    printf '%s' "$listing" | grep -Fq "\"$1\""
}

# --------------------------------------------------------------------------------------
# Project
# --------------------------------------------------------------------------------------

cd "$REPO_ROOT"

say "resolving the project"
if railway status --json >/dev/null 2>&1; then
    linked="$(railway status --json 2>/dev/null | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin).get("name", "?"))
except Exception:
    print("?")
')"
    info "already linked to project: ${linked}"
else
    info "not linked; creating project \"${PROJECT_NAME}\""
    railway init --name "$PROJECT_NAME" --json >/dev/null
    info "created and linked \"${PROJECT_NAME}\""
fi

# --------------------------------------------------------------------------------------
# Qdrant service
# --------------------------------------------------------------------------------------

say "Qdrant service (${QDRANT_SERVICE})"
if service_exists "$QDRANT_SERVICE"; then
    info "already exists, leaving it alone"
else
    railway add --service "$QDRANT_SERVICE" --image "$QDRANT_IMAGE" --json >/dev/null
    info "created from ${QDRANT_IMAGE}"
fi

if volume_mounted "$QDRANT_MOUNT"; then
    info "volume at ${QDRANT_MOUNT} already present"
else
    # `--service` is an option of `railway volume`, NOT of its `add` subcommand: putting it
    # after `add` fails with "unexpected argument '--service' found" (railway 4.66.1).
    railway volume --service "$QDRANT_SERVICE" add --mount-path "$QDRANT_MOUNT" --json >/dev/null
    info "volume attached at ${QDRANT_MOUNT}"
fi

# Railway's private network is IPv6. Qdrant's default bind is 0.0.0.0, which is IPv4 only,
# so without this the API service resolves qdrant.railway.internal and then hangs on a
# connection that nothing is listening for. Linux dual-stack sockets accept IPv4 on the
# same socket, so :: loses nothing.
#
# ON_DISK_PAYLOAD keeps the passage text off the heap -- it is the bulk of the bytes and is
# only read for the final top-k. HNSW_INDEX__ON_DISK stays false so graph traversal never
# takes a page fault: measured on the local 2.27M-point index, on-disk gave p50 6.8ms but
# p90 264ms and p100 302ms, which blew the per-stage timeout and silently degraded the
# system to lexical-only retrieval.
#
# Note what is deliberately NOT copied from docker-compose.yml: MEMMAP_THRESHOLD_KB. That
# exists locally to survive a 2.27M-chunk index on a 24GB laptop. Setting it here would
# memory-map the vectors and reintroduce exactly the tail the deploy-sized corpus exists to
# avoid.
say "Qdrant settings"
railway variable set \
    "QDRANT__SERVICE__HOST=::" \
    "QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=64" \
    "QDRANT__STORAGE__ON_DISK_PAYLOAD=true" \
    "QDRANT__STORAGE__HNSW_INDEX__ON_DISK=false" \
    "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=4" \
    "QDRANT__LOG_LEVEL=WARN" \
    --service "$QDRANT_SERVICE" >/dev/null
info "IPv6 bind + in-RAM HNSW + on-disk payload set"

# --------------------------------------------------------------------------------------
# API service
# --------------------------------------------------------------------------------------

say "API service (${API_SERVICE})"
if service_exists "$API_SERVICE"; then
    info "already exists, leaving it alone"
else
    railway add --service "$API_SERVICE" --json >/dev/null
    info "created (empty; \`railway up\` uploads this directory into it)"
fi

if volume_mounted "$API_MOUNT"; then
    info "volume at ${API_MOUNT} already present"
else
    railway volume --service "$API_SERVICE" add --mount-path "$API_MOUNT" --json >/dev/null
    info "volume attached at ${API_MOUNT}"
    warn "check its size in the dashboard before deploying -- the first boot writes the"
    warn "HuggingFace parquet cache, docs.jsonl and the BM25 index here."
fi

# `${{service.VAR}}` is Railway's own interpolation syntax, resolved server-side at deploy
# time, so it survives the service being redeployed onto a different host. Built with
# printf from a single-quoted format string so bash never tries to expand it.
QDRANT_URL="$(printf 'http://${{%s.RAILWAY_PRIVATE_DOMAIN}}:%s' "$QDRANT_SERVICE" "$QDRANT_PORT")"

# --skip-deploys: this service has nothing deployed yet on a first run, and on a re-run the
# `railway up` at the end is what should pick the values up -- one deploy, not six.
say "API settings"
railway variable set \
    "VOXRAG_QDRANT_URL=${QDRANT_URL}" \
    "VOXRAG_VECTOR_STORE=qdrant" \
    "VOXRAG_DEFAULT_MODE=fast" \
    "PORT=${API_PORT}" \
    --service "$API_SERVICE" --skip-deploys >/dev/null
info "VOXRAG_QDRANT_URL -> ${QDRANT_URL}"
info "PORT=${API_PORT} (matches EXPOSE in the Dockerfile; tells the edge proxy where to go)"

# --------------------------------------------------------------------------------------
# Sarvam key -- the one genuinely required secret
# --------------------------------------------------------------------------------------

say "Sarvam STT key"
sarvam_key="${VOXRAG_SARVAM_API_KEY:-}"
if [[ -z "$sarvam_key" && -f "${REPO_ROOT}/.env" ]]; then
    # First assignment only. Trailing CR (CRLF files) and surrounding quotes are stripped
    # in bash rather than piped through tr, so the value never reaches another process.
    sarvam_key="$(sed -n 's/^[[:space:]]*VOXRAG_SARVAM_API_KEY[[:space:]]*=[[:space:]]*//p' \
        "${REPO_ROOT}/.env" | head -n1)"
    sarvam_key="${sarvam_key%$'\r'}"
    sarvam_key="${sarvam_key#\"}"; sarvam_key="${sarvam_key%\"}"
    sarvam_key="${sarvam_key#\'}"; sarvam_key="${sarvam_key%\'}"
fi

if [[ -n "$sarvam_key" ]]; then
    # --stdin keeps the value out of argv, out of shell history and out of `ps`.
    if printf '%s' "$sarvam_key" \
        | railway variable set VOXRAG_SARVAM_API_KEY --stdin \
              --service "$API_SERVICE" --skip-deploys >/dev/null 2>&1; then
        sarvam_key=""
        info "VOXRAG_SARVAM_API_KEY set (value not printed anywhere)"
    else
        sarvam_key=""
        die "failed to set VOXRAG_SARVAM_API_KEY (the value was not shown; update the
      railway CLI and re-run, or set it by hand in the dashboard)"
    fi
else
    warn "no key found in \$VOXRAG_SARVAM_API_KEY or ./.env"
    warn "voice input will return empty transcripts until you set it -- the image carries no"
    warn "local Whisper fallback. Text queries work regardless. Set it with:"
    # Prompted rather than typed inline: an interactive shell records the whole pipeline in
    # its history, so a literal key on the command line outlives the command.
    warn "  read -rsp 'Sarvam key: ' K && echo"
    warn "  printf '%s' \"\$K\" | railway variable set VOXRAG_SARVAM_API_KEY --stdin --service ${API_SERVICE}; unset K"
fi

# --------------------------------------------------------------------------------------
# Public domain
# --------------------------------------------------------------------------------------

say "public domain for ${API_SERVICE}"
# Railway allows exactly one generated domain per service, so a re-run fails here. That is
# not an error condition for this script.
if domain_out="$(railway domain --service "$API_SERVICE" --port "$API_PORT" 2>&1)"; then
    printf '%s\n' "$domain_out" | sed 's/^/    /'
else
    info "already has a domain (or none could be generated); check: railway open"
fi

# --------------------------------------------------------------------------------------
# Next steps
# --------------------------------------------------------------------------------------

say "provisioning done. Nothing has been deployed yet."
cat <<EOF

  Next:

    1. Deploy the API service (build takes a few minutes):
         railway up --service ${API_SERVICE}

    2. Watch the first-boot index build (it runs in the background; the port is already
       open, so the deploy goes healthy long before the index is ready):
         railway logs --service ${API_SERVICE}

       Expect: "[entrypoint] building index: langs=as,bn,... limit=20000 strategy=parent_child"
       then, tens of minutes later, "[entrypoint] index build complete".

    3. Verify:
         curl -s https://<your-domain>/api/health | python3 -m json.tool

       Healthy means ok=true AND vector_store reporting a non-zero vector count.

  Full walkthrough: the Deploy section of the README.
EOF
