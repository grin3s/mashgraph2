import glob
from multiprocessing import Pool
import os


def async_run(all_files, out_folder):
    for f in all_files:
        filename = f.split("/")[-1]
        print("processing {0}".format(f))
        out_path = out_folder + filename
        os.system("python3.5 run_all.py {0} {1}".format(f, out_path))
        print("finished {0}".format(f))
    return None

if __name__ == '__main__':
    n_processes = 2
    in_folder = "../training/"
    out_folder = "../results2/"

    files = glob.glob(in_folder + "*.tif")
    n_steps_per_process = int(len(files) / n_processes) + 1
    files_chunks = [files[i: i + n_steps_per_process] for i in range(0, len(files), n_steps_per_process)]

    pool = Pool(processes=n_processes)

    proc_results = []

    for chunk in files_chunks:
        res = pool.apply_async(async_run, (chunk, out_folder))
        proc_results.append(res)

    for res in proc_results:
        res.get()

