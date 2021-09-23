#!/bin/bash
# Pulls from remote.

FOLDER=pa4

scp -r jjjlaw@csl2wk37.cse.ust.hk:/homes/jjjlaw/courses/comp4901q/"$FOLDER"/ .
find "$FOLDER" -type f \( -name '*.h' -o -name '*.cpp' -o -name '*.cu' \) -exec mv {} . \;
rm -rf "$FOLDER"
