# Single poisoners (We could also change eval_grid from 5 to 1)
python code/ML_main.py mnist 3000 1_1_7 
python code/ML_main.py mnist 3000 1_4_9
python code/ML_main.py mnist 3000 1_0_8

# 99% poisoners !! Run on lev
python code/ML_main.py mnist 1000 90_1_7

# 5x5
python code/ML_main.py mnist 300 5_1_7 5_4_9 5_0_8 5_3_8 5_6_8
