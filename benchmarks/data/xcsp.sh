#!/bin/bash

mkdir -p XCSP
cd XCSP

wget https://www.cril.univ-artois.fr/~lecoutre/compets/instances$1.zip
unzip instances$1.zip
rm instances$1.zip

# Special for XCSP23
mv XCSP23_V2 XCSP23
mv instances$1 $1

find $1 -type d -exec chmod u+w {} +

find $1 -type f -name '*.lzma' -exec chmod u+w {} +

find "$1" -type f -name '*.lzma' -exec sh -c '
  for file do
    xz --decompress "$file"
  done
' sh {} +
