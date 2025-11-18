from multiprocessing import Pool
from pathlib import Path
# Also input nnUNetV2 preprocessing function
# placeholder: preprocess_function(input_path, output_dir)


def multiproc_run():
    src_dir = Path("/PC_Classifier/nnUNet_raw") # Path to raw data
    dst_dir = Path("/PC_Classifier/nnUNet_preprocessed") # Path to destination directory

    case_files = list(sorted(src_dir.glob("*.nii.gz")))

    # build a list of argument pairs
    tasks = [(str(f), str(dst_dir)) for f in case_files]

    with Pool(processes=16) as pool: # adjust processes based on resource availability
        pool.starmap(preprocess_function, tasks)



if __name__ == "__main__":
    multiproc_run()