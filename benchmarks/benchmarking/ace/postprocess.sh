#!/bin/bash

set -x

# Check if a directory was provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <campaign-directory>"
    exit 1
fi

dir=$(dirname $0)
python3 $dir/postprocess.py $1
