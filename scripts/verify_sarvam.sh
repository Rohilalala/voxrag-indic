#!/usr/bin/env bash
# Ground-truth check of the Sarvam speech-to-text contract, verified working 2026-08-15.
#
# This exists because the STT wire format is the one part of the pipeline that cannot be
# unit-tested against a local fixture -- if the endpoint, the auth header name, the model id
# or the language code is wrong, the demo dies live and the error messages are unhelpful.
# Run it after any change to stt/sarvam.py, and before recording the demo video.
#
# Verified response shape:
#   {"request_id": str, "transcript": str, "language_code": "hi-IN",
#    "timestamps": {...}?, "language_probability": float?}

set -euo pipefail

cd "$(dirname "$0")/.."
[[ -f .env ]] || { echo "no .env -- set VOXRAG_SARVAM_API_KEY first"; exit 1; }
set -a; . ./.env; set +a
: "${VOXRAG_SARVAM_API_KEY:?VOXRAG_SARVAM_API_KEY is not set in .env}"

MODEL="${VOXRAG_SARVAM_MODEL:-saaras:v3}"
OUT_DIR="${1:-data/audio}"
mkdir -p "${OUT_DIR}"

# macOS ships Indic TTS voices, which gives us free, reproducible, correctly-scripted test
# audio in three of the indexed languages -- no recording, no committed binary fixtures.
declare -a CASES=(
    "hi|hi-IN|Lekha|भारत की राजधानी नई दिल्ली है"
    "bn|bn-IN|Piya|ভারতের রাজধানী নয়াদিল্লি"
    "ta|ta-IN|Vani|இந்தியாவின் தலைநகரம் புது தில்லி"
)

fail=0
for case in "${CASES[@]}"; do
    IFS='|' read -r lang code voice text <<<"${case}"
    clip="${OUT_DIR}/${lang}_test.aiff"

    if ! say -v "${voice}" -o "${clip}" "${text}" 2>/dev/null; then
        echo "SKIP ${lang}: voice '${voice}' not installed (System Settings > Spoken Content)"
        continue
    fi

    started=$(date +%s.%N)
    body=$(curl -s --max-time 45 -X POST "${VOXRAG_SARVAM_BASE_URL:-https://api.sarvam.ai}/speech-to-text" \
        -H "api-subscription-key: ${VOXRAG_SARVAM_API_KEY}" \
        -F "file=@${clip}" -F "model=${MODEL}" -F "language_code=${code}" || true)
    elapsed=$(echo "$(date +%s.%N) - ${started}" | bc)

    got=$(printf '%s' "${body}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("transcript",""))' 2>/dev/null || echo "")
    if [[ -z "${got}" ]]; then
        # Truncate: an auth error body can be long, and it must never echo the key back.
        echo "FAIL ${lang}: no transcript -- ${body:0:200}"
        fail=1
    else
        printf 'PASS %s  %5.2fs  sent=%s\n           got=%s\n' "${lang}" "${elapsed}" "${text}" "${got}"
    fi
done

echo
echo "NOTE: these round trips are ~1.5s each. That is why the 200ms budget is measured on"
echo "CORE (total minus STT) -- a vendor network call cannot be optimised into 200ms."
exit "${fail}"
