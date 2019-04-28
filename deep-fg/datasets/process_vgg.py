import pandas as pd
import os
import pdb
import shutil 
import numpy as np
"""
Moves the folders with the top k sample nums to datadir + data/
"""
k = 10

datadir = "/bigdata/vggface2"
meta_path = os.path.join(datadir, "identity_meta.csv")

meta_csv = pd.read_csv(meta_path, sep=", ")
meta_csv = meta_csv[meta_csv['Flag'] == 1]
sample_nums = meta_csv['Sample_Num'].sort_values(ascending=False)
topk = sample_nums.iloc[:k].reset_index()
topk_indices = topk['index']

topk_csv = meta_csv.loc[topk_indices].reset_index()
topk_ids = topk_csv['Class_ID']

# Copy folders to datadir
for class_id in topk_ids:
    source_dir = os.path.join(datadir, "vggface2_train", "train", class_id)
    dest_dir = os.path.join(datadir, "data", class_id)
    if not os.path.isdir(dest_dir):
        shutil.copytree(source_dir, dest_dir)

csv_path = os.path.join(datadir, "top{}.csv".format(k))
topk_csv.to_csv(csv_path, index=False)

topk_files_df = pd.DataFrame()

data = [] 
# Create a new meta file with images mapped to identities
class_idx = 0
for class_id in topk_ids:
    dest_dir = os.path.join(datadir, "data", class_id)
    class_meta = topk_csv[topk_csv['Class_ID']==class_id].values.tolist()[0]
    for root, dirs, files in os.walk(dest_dir):
        for file_name in files:
            row = class_meta.copy()
            row.append(file_name)
            row.append(class_idx)
            data.append(row)
    class_idx += 1

topk_files_df = pd.DataFrame(data, columns=['index', 'Class_ID', 'Name', 'Sample_Num', 'Flag', 'Gender', 'file', 'idx'])

train_flag = np.array([]).astype('int')
valid_size = 0.3
for i in range(len(topk_ids)):
    df = topk_files_df[topk_files_df['idx'] == i]
    num_train = len(df)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    c_train_flag = np.zeros(num_train).astype('int')
    c_train_flag[valid_idx] = 1
    train_flag = np.concatenate((train_flag, c_train_flag))

topk_files_df['train_flag'] = train_flag

topk_files_csv_path = os.path.join(datadir, "top{}_files.csv".format(k))
topk_files_df.to_csv(topk_files_csv_path, index=False)
pdb.set_trace()
