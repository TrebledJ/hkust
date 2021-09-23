#!/bin/bash
# Pushes to remote.

FOLDER=pa3
FILES=(../utils/*.h *.h *.cpp *hostfile)

scp ${FILES[@]} jjjlaw@csl2wk37.cse.ust.hk:/homes/jjjlaw/courses/comp4901q/"$FOLDER"
