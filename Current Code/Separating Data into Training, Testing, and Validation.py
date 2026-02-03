import glob
import random
import os
import csv

monthly_files = glob.glob("../GHAP_PM2.5_M1K_*.nc")

print("Total monthly files:", len(monthly_files))

if len(monthly_files) == 0:
    raise RuntimeError("No monthly .nc files found. Check the path.")

random.seed(42)
random.shuffle(monthly_files)

n = len(monthly_files)
n_train = int(0.7 * n)
n_test = int(0.2 * n)

train_files = monthly_files[:n_train]
test_files = monthly_files[n_train:n_train + n_test]
val_files = monthly_files[n_train + n_test:]

print("Train:", len(train_files))
print("Test:", len(test_files))
print("Validation:", len(val_files))

def parse_monthly_filename(path):
    fname = os.path.basename(path)
    date_str = fname.split("_")[3]
    year = date_str[:4]
    month = date_str[4:6]
    return year, month, path

def save_split_table(file_list, out_name):
    with open(out_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index", "Year", "Month", "File Path"])
        for idx, f in enumerate(file_list, start=1):
            year, month, path = parse_monthly_filename(f)
            writer.writerow([idx, year, month, path])

save_split_table(train_files, "train_files.csv") #50 .nc files
save_split_table(test_files, "test_files.csv") #14 .nc files
save_split_table(val_files, "val_files.csv") #8 .nc files

print("Split tables saved successfully.")
