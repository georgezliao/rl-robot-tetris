#!/usr/bin/env bash
# Print the JDK home used to run the JavaFX bridge. run_java_bridge.sh and the
# run-provenance record both call this so they report the same JDK.
set -euo pipefail

if [[ -n "${JAVAFX_JAVA_HOME:-}" ]]; then
  echo "$JAVAFX_JAVA_HOME"
elif [[ -d "/Library/Java/JavaVirtualMachines/liberica-jdk-8-full.jdk/Contents/Home" ]]; then
  echo "/Library/Java/JavaVirtualMachines/liberica-jdk-8-full.jdk/Contents/Home"
else
  /usr/libexec/java_home -v 1.8
fi
