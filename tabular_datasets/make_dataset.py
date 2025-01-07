# Generate satimage dataset

import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from make_utils import CATEGORICAL, CONTINUOUS, ORDINAL, verify

def project_table(data, meta):
    values = np.zeros(shape=data.shape, dtype='float32')

    for id_, info in enumerate(meta):
        if info['type'] == CONTINUOUS:
            values[:, id_] = data.iloc[:, id_].values.astype('float32')
        else:
            mapper = dict([(item, id) for id, item in enumerate(info['i2s'])])
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
            values[:, id_] = mapped
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
    return values

def create_dataset(output_dir, temp_dir, name, col_type, csv_path, csv_delimiter, num_datapoints=-1):
    try:
        os.mkdir(output_dir) 
    except:
        pass

    try:
        os.mkdir(temp_dir)
    except:
        pass

    df = pd.read_csv(csv_path, dtype='str', delimiter=csv_delimiter)
    df = pd.DataFrame(df)

    if num_datapoints > 0 and num_datapoints <= len(df):
        df = df.sample(n=num_datapoints, replace=False)


    print(df.shape)

    meta = []
    for id_, info in enumerate(col_type):
        if info[1] == CONTINUOUS:
            meta.append({
                "name": info[0],
                "type": info[1],
                "min": np.min(df.iloc[:, id_].values.astype('float')),
                "max": np.max(df.iloc[:, id_].values.astype('float'))
            })
        else:
            if info[1] == CATEGORICAL:
                value_count = list(dict(df.iloc[:, id_].value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
            else:
                mapper = info[2]

            meta.append({
                "name": info[0],
                "type": info[1],
                "size": len(mapper),
                "i2s": mapper
            })

    tdata = project_table(df, meta)

    config = {
                'columns':meta, 
                'problem_type':'binary_classification'
            }

    np.random.seed(0)
    np.random.shuffle(tdata)

    train_ratio = int(tdata.shape[0]*0.2)
    t_train = tdata[:-train_ratio]
    t_test = tdata[-train_ratio:]

    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, test=t_test)

    verify("{}/{}.npz".format(output_dir, name),
            "{}/{}.json".format(output_dir, name))

def main():
    parser = argparse.ArgumentParser(description="generate a tabular dataset.")
    parser.add_argument("--name", type=str, required=True, help="name for generated files.")
    parser.add_argument("--csv_path", type=str, required=True, help="path where the CSV file is stored.")
    parser.add_argument("--col_type_file", type=str, required=True, help="path to a JSON file defining column types.")
    parser.add_argument("--output_dir", type=str, default=".", help="directory to output created dataset files to.")
    parser.add_argument("--temp_dir", type=str, default="./tmp", help="directory to place temporary files.")
    parser.add_argument("--csv_delimiter", type=str, default=",", help="delimiter for CSV dataset.")
    parser.add_argument("--num_datapoints", type=int, default=-1, help="number of datapoints to randomly sample (-1 for full dataset).")

    args = parser.parse_args()

    name = args.name
    csv_path = Path(args.csv_path).resolve()
    col_type_file = Path(args.col_type_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    temp_dir = Path(args.output_dir).resolve()
    csv_delimiter = args.csv_delimiter
    num_datapoints = args.num_datapoints

    try:
        with open(col_type_file, "r") as f:
            col_type = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {col_type_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {col_type_file}: {e}")
    except PermissionError:
        raise PermissionError(f"Permission denied when accessing file: {col_type_file}")

    create_dataset(output_dir, temp_dir, name, col_type, csv_path, csv_delimiter, num_datapoints=num_datapoints)

if __name__ == "__main__":
    main()