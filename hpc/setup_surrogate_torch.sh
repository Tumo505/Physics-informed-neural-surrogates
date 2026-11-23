#!/bin/bash
# Called inside the allocated PBS job. Keep FE Python untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
python_seed="${SURROGATE_PYTHON:-$PWD/.hpc/native-v2/bin/python}"
environment="$PWD/.hpc/surrogate-ml"
if [ ! -x "$environment/bin/python" ]; then
  "$python_seed" -m venv "$environment"
fi
"$environment/bin/python" -m pip install --upgrade pip
# cu118 retains Maxwell support. The PBS worker also checks a real graph backward
# pass against the installed driver, and fails clearly if it is incompatible.
"$environment/bin/python" -m pip install --only-binary=:all: torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
"$environment/bin/python" -m pip install --only-binary=:all: numpy==1.26.4 scipy==1.14.1 matplotlib==3.9.4
"$environment/bin/python" -m pip check
"$environment/bin/python" -m pip freeze > "$environment/requirements-installed.txt"
touch "$environment/.installed"
