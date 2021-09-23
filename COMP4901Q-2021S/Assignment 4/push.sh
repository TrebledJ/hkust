#!/bin/bash
# Pushes to remote.

FOLDER=pa4
FILES=(../utils/*.h ../utils/*.cpp *.h *.cu *.cpp *hostfile)
scp ${FILES[@]} jjjlaw@csl2wk37.cse.ust.hk:/homes/jjjlaw/courses/comp4901q/"$FOLDER"
