#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build/java"

JAVA_FX_HOME="$("$PROJECT_DIR/scripts/java_home.sh")"

mkdir -p "$BUILD_DIR"
"$JAVA_FX_HOME/bin/javac" -d "$BUILD_DIR" "$PROJECT_DIR"/tetris/*.java
