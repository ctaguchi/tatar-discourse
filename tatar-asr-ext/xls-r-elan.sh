#!/bin/bash
#
# Set environmental variables and locale-specific settings needed for
# this recognizer to run as expected before calling the recognizer.

# Change the environment variable below to the location of the ffmpeg
# in your system.
export FFMPEG_DIR="/opt/homebrew/bin/ffmpeg"

export LC_ALL="en_US.UTF-8"
export PYTHONIOENCODING="utf-8"
export PATH="$PATH:$FFMPEG_DIR"

# Run
exec poetry run python elan.py