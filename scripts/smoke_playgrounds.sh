#!/usr/bin/env bash
# Exits non-zero if any playground endpoint fails the scripted moments.
set -euo pipefail
BASE="${BASE:-http://localhost:8000}"

echo "Health..."
curl -fsS "$BASE/health" | jq '.playgrounds'

echo "Words: jazz neighbors"
curl -fsS "$BASE/playgrounds/words/query?q=jazz" | jq '.results[0].label'

echo "ChEMBL: aspirin neighbors"
curl -fsS "$BASE/playgrounds/chembl/query?q=CC(=O)Oc1ccccc1C(=O)O" | jq '.results[0].label'

echo "Spike: top neighbor of Wuhan-Hu-1"
curl -fsS "$BASE/playgrounds/spike/query?q=Wuhan-Hu-1" | jq '.results[0].label' || true

echo "Proteins: gallery (size)"
curl -fsS "$BASE/playgrounds/proteins/gallery" | jq '.items | length'

echo "All endpoints OK."
