#!/usr/bin/env bash
# Parallel chunked downloader for the MSMARCO-XI parquet files.
#
# Why this exists instead of `hf_hub_download`:
#   1. hf_hub_download wedges on this repo -- observed parking at an exact chunk boundary
#      with live sockets, ~0% CPU and no throughput, for tens of minutes.
#   2. The CDN throttles PER CONNECTION, not per client. Measured: one stream 180 KB/s,
#      four concurrent range requests ~117 KB/s each (~470 KB/s aggregate). So splitting a
#      file into N ranges is a ~4-8x speedup, not a rounding error.
#
# Usage: scripts/fetch_parquet.sh [outdir] [conns] [file...]
#   scripts/fetch_parquet.sh data/parquet 16 hinval tamval benval

set -euo pipefail

# SPLIT_DIR selects validation (the eval set) or train. Kept an env var rather than a
# positional so existing invocations keep working unchanged.
SPLIT_DIR="${SPLIT_DIR:-validation}"
REPO_URL="https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/${SPLIT_DIR}"
OUT_DIR="${1:-data/parquet}"
CONNS="${2:-16}"
shift 2 2>/dev/null || true
FILES=("${@:-hinval tamval benval}")
# shellcheck disable=SC2206
[[ ${#FILES[@]} -eq 1 ]] && FILES=(${FILES[0]})

mkdir -p "${OUT_DIR}"

for stem in "${FILES[@]}"; do
    url="${REPO_URL}/${stem}.parquet"
    final="${OUT_DIR}/${stem}.parquet"

    total=$(curl -sIL "${url}" | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {v=$2} END{gsub(/\r/,"",v); print v}')
    if [[ -z "${total}" || "${total}" -lt 1000 ]]; then
        echo "!! ${stem}: could not determine size (got '${total}')" >&2
        exit 1
    fi

    if [[ -f "${final}" ]] && [[ "$(stat -f%z "${final}" 2>/dev/null || echo 0)" == "${total}" ]]; then
        echo "== ${stem}: already complete ($((total / 1048576)) MB)"
        continue
    fi

    echo "== ${stem}: $((total / 1048576)) MB via ${CONNS} connections"
    part_dir="${OUT_DIR}/.parts-${stem}"
    mkdir -p "${part_dir}"

    chunk=$(( (total + CONNS - 1) / CONNS ))
    pids=()
    for ((i = 0; i < CONNS; i++)); do
        start=$((i * chunk))
        end=$((start + chunk - 1))
        [[ ${end} -ge ${total} ]] && end=$((total - 1))
        [[ ${start} -ge ${total} ]] && break
        part="${part_dir}/part.$(printf '%03d' "${i}")"
        # -C - resumes a partially fetched chunk, so re-running after a drop is cheap.
        # --retry-all-errors covers the CDN's occasional 5xx mid-transfer.
        curl -sL -C - --retry 5 --retry-delay 2 --retry-all-errors \
            -r "${start}-${end}" -o "${part}" "${url}" &
        pids+=($!)
    done

    fail=0
    for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
    [[ ${fail} -eq 0 ]] || { echo "!! ${stem}: a chunk failed; re-run to resume" >&2; exit 1; }

    # Concatenate in lexical order -- the %03d padding above is what makes that correct.
    cat "${part_dir}"/part.* > "${final}"
    got=$(stat -f%z "${final}" 2>/dev/null || echo 0)
    if [[ "${got}" != "${total}" ]]; then
        echo "!! ${stem}: size mismatch got=${got} want=${total}; parts kept for retry" >&2
        exit 1
    fi
    rm -rf "${part_dir}"

    # Parquet's magic number is "PAR1" at both ends. Checking the trailer catches a
    # truncated or mis-ordered concatenation before pyarrow reports something cryptic.
    head -c 4 "${final}" | grep -q PAR1 && tail -c 4 "${final}" | grep -q PAR1 \
        && echo "   ok $((got / 1048576)) MB, PAR1 header+trailer verified" \
        || { echo "!! ${stem}: not a valid parquet"; exit 1; }
done

echo "ALL-PARQUET-DONE"
