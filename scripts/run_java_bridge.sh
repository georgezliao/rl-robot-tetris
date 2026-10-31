#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_DIR/scripts/build_java_bridge.sh"

JAVA_FX_HOME="$("$PROJECT_DIR/scripts/java_home.sh")"

# TETRIS_BRIDGE_OFFSCREEN=1 keeps the game window outside the visible desktop.
# Observations are rendered from the scene graph, so they do not change.
JAVA_OPTIONS=()
if [[ "${TETRIS_BRIDGE_OFFSCREEN:-0}" == "1" ]]; then
  JAVA_OPTIONS+=("-Dtetris.bridge.offscreen=true")
fi

exec "$JAVA_FX_HOME/bin/java" "${JAVA_OPTIONS[@]+"${JAVA_OPTIONS[@]}"}" \
  -cp "$PROJECT_DIR/build/java" tetris.RlBridge
