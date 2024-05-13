#!/bin/bash

set -x

# Check if a directory was provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <campaign-directory>"
    exit 1
fi


python3 postprocess.py $1

