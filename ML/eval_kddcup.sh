# Single poisoners (We could also change eval_grid from 5 to 1)
python code/ML_main.py kddcup 3000 1_1_7
python code/ML_main.py kddcup 3000 1_4_9
python code/ML_main.py kddcup 3000 1_0_8
python code/ML_main.py kddcup 3000 1_6_11

#1x5
python code/ML_main.py kddcup 3000 5_6_11

#5x5
python code/ML_main.py kddcup 300 5_0_11 5_3_13 5_8_7 5_17_18 5_21_15

#2x5
python code/ML_main.py kddcup 300 5_0_11 5_3_13

#0_11, 2_11, 3_11, 5_11, 13_11, 15_11
python code/ML_main.py kddcup 300 5_0_11 5_2_11 5_3_11 5_13_11 5_20_11
