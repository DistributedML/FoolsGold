#!/bin/bash

ARG="$1"' 3000'
#echo $ARG
#python code/ML_main.py $ARG

for i in `seq 4 9`;
do
  for j in `seq 0 9`;
  do
    if [ $i -eq $j ]
    then
        echo "i and j equal"
    else
        NEWARG="$ARG"' 5_'"$i"'_'"$j"
        FILENAME="autologs/"$NEWARG'.log'
        echo $FILENAME
        python code/ML_main.py $NEWARG > "$FILENAME"
    fi
  done
done
