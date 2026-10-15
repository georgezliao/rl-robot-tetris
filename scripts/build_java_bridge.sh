#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build/java"

if [[ -n "${JAVAFX_JAVA_HOME:-}" ]]; then
  JAVA_FX_HOME="$JAVAFX_JAVA_HOME"
elif [[ -d "/Library/Java/JavaVirtualMachines/liberica-jdk-8-full.jdk/Contents/Home" ]]; then
  JAVA_FX_HOME="/Library/Java/JavaVirtualMachines/liberica-jdk-8-full.jdk/Contents/Home"
else
  JAVA_FX_HOME="$(/usr/libexec/java_home -v 1.8)"
fi

mkdir -p "$BUILD_DIR"
"$JAVA_FX_HOME/bin/javac" -d "$BUILD_DIR" "$PROJECT_DIR"/tetris/*.java
