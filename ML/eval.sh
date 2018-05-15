#!/bin/bash

ARG="$1"
#echo $ARG
#python code/ML_main.py $ARG

for i in `seq 0 1`;
do
  NEWARG=$ARG' '"$i"'17'
  for j in `seq 0 1`;
  do
    NEWARG2=$NEWARG' '"$j"'49'
    FILENAME=$NEWARG2'.log'
    python code/ML_main.py $NEWARG2 > "$FILENAME" &
  done
done
