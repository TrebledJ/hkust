#!/bin/bash
# Pulls from remote.

scp -r jjjlaw@csl2wk37.cse.ust.hk:/homes/jjjlaw/courses/comp4901q/asgn3/ .
find asgn3 -type f \( -name '*.h' -o -name '*.cpp' \) -exec mv {} . \;
rm -rf asgn3
