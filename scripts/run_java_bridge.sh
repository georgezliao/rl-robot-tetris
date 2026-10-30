#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_DIR/scripts/build_java_bridge.sh"

JAVA_FX_HOME="$("$PROJECT_DIR/scripts/java_home.sh")"

exec "$JAVA_FX_HOME/bin/java" -cp "$PROJECT_DIR/build/java" tetris.RlBridge
