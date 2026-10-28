#!/usr/bin/env bash
# Full pipeline: data -> graph stats -> train -> evaluate -> tests. Run from the project root.
set -euo pipefail
PY=${PY:-python}
$PY -m src.data            # downloads real data on first run
$PY -m src.graph
$PY -m src.train --model all
$PY -m src.evaluate
$PY -m pytest -q tests
