# Single poisoners (We could also change eval_grid from 5 to 1)
python code/ML_main.py amazon 30 1_10_15
python code/ML_main.py amazon 30 1_5_10
python code/ML_main.py amazon 30 1_30_35
python code/ML_main.py amazon 30 1_40_20

#1x5
python code/ML_main.py amazon 10 5_30_35

#5x5
python code/ML_main.py amazon 100 5_0_10 5_5_25 5_20_40 5_35_10 5_20_15

#2x5
python code/ML_main.py amazon 100 5_0_10 5_5_25

#0_11, 2_11, 3_11, 5_11, 13_11, 15_11
python code/ML_main.py amazon 5 5_0_40 5_5_40 5_10_40 5_20_40 5_25_40
