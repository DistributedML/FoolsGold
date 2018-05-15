#!/bin/bash

ARG="$1"

for i in `seq 0 1`;
do
  NEWARG=$ARG' 3000 '"$i"'_1_7'
  for j in `seq 0 1`;
  do
    NEWARG2=$NEWARG' 3000 '"$j"'_4_9'
    FILENAME=$NEWARG2'.log'
    python code/ML_main.py $NEWARG2 > "$FILENAME" &
  done
done
