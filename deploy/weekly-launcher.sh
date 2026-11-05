#!/bin/sh
set -eu
# The explicitly selected operator env file is trusted shell configuration.
# Never accept its location from a lesson or other untrusted document.
: "${SMART_AGENTS_ENV_FILE:?Set a private operator environment file}"
set -a
. "$SMART_AGENTS_ENV_FILE"
set +a
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo"
exec "$repo/.venv/bin/smart-agents" weekly "$@"
