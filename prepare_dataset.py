import os

from glob import glob
import pandas as pd
import typer
from pandarallel import pandarallel

from bench_xecg.dataset.dataset_preparation_utils import *

app = typer.Typer(help="Clean and prepare ECG datasets")

## example usage
# start srun

# code 15 filtering with 10 seconds:
# python prepare_dataset.py --data-folder '/scratch/datasets/CODE15/processed/' --dataset code15 --num-workers 32 --label-file-output exams_filtered_10sec --filter-10seconds

# chapman ningbo filtering with 10 seconds:
# python prepare_dataset.py --data-folder '/home/datasets/ChapmanNingbo/1.0.0/' --dataset chapman --num-workers 32 --label-file-output exams_filtered_10sec --filter-10seconds


def find_records(folder: str, file_extension: str = ".hea"):
    records = set()
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(file_extension):
                record = os.path.relpath(
                    os.path.join(root, file), folder
                )[:-len(file_extension)]
                records.add(record)
    return sorted(records)

@app.command()
def main(
    dataset: str = typer.Option(
        ..., help="Dataset name: mimic, code, code15, ptbxl, chapman, heedb"
    ),
    data_folder: str = typer.Option(
        "/media/Volume/data/MIMIC_IV/",
        help="Path to raw data folder",
    ),
    label_file: str = typer.Option(
        "/media/Volume/data/MIMIC_IV/records_w_diag_icd10.csv",
        help="Path to the label file",
    ),
    label_file_output: str = typer.Option(
        "exams_filtered",
        help="Output CSV name (without extension)",
    ),
    filter_10seconds: bool = typer.Option(
        False,
        "--filter-10seconds",
        help="Filter exams shorter than 10 seconds",
    ),
    check_variance: bool = typer.Option(
        False,
        "--check-variance",
        help="Filter exams shorter than 10 seconds",
    ),
    num_workers: int = typer.Option(
        16,
        help="Number of parallel workers",
    ),
):
    pandarallel.initialize(
        progress_bar=False,
        verbose=0,
        nb_workers=num_workers,
    )

    typer.echo(f"Running with config:\n"
               f"dataset={dataset}, data_folder={data_folder}, "
               f"num_workers={num_workers}, filter_10seconds={filter_10seconds}")

    # --------------------
    # Load exams
    # --------------------
    if dataset == "chapman":
        with open(os.path.join(data_folder, "RECORDS"), "r") as f:
            paths = f.readlines()

        all_files = []
        for path in paths:
            with open(
                os.path.join(data_folder, path.strip(), "RECORDS"), "r"
            ) as f:
                lines = f.readlines()
            all_files += [
                os.path.join(path.strip(), line.strip()) for line in lines
            ]

        exams = pd.DataFrame({"file_name": all_files})

    elif dataset == "code":
        with open(os.path.join(data_folder, "RECORDS.txt"), "r") as f:
            paths = [line.strip() for line in f.readlines()]

        exams = pd.DataFrame({"file_name": paths})

    elif dataset == "code15":
        # if code15 first transforem hdf5 to wfdb and save in a temp folder, then read the hea files
        exams = pd.read_csv(os.path.join(data_folder, 'raw', "exams.csv"))

        # check if processed folder exists, if not create it and process the files
        if not os.path.exists(os.path.join(data_folder, "processed")):
            os.makedirs(os.path.join(data_folder, 'processed'), exist_ok=True)

            exams.parallel_apply(
                lambda row: save_record_hdf5_to_wfdb(
                    os.path.join(data_folder, 'raw', row['trace_file']),
                    row['exam_id'],
                    os.path.join(data_folder, 'processed')
                ),
                axis=1,
            )

        # add header filenames to exams
        exams["file_name"] = exams.parallel_apply(
            lambda row: os.path.join("processed", str(row['exam_id'])),
            axis=1,
        )

        print("DF CODE15% head:")
        print(exams.head())
        
    elif dataset == "heedb":
        all_records = find_records(data_folder, file_extension=".hea")
        typer.echo(f"Found records: {len(all_records)}")
        exams = pd.DataFrame({"file_name": all_records})

    else:
        exams = pd.read_csv(label_file)

    typer.echo(f"Initial rows: {len(exams)}")

    # --------------------
    # Validation
    # --------------------
    if dataset == "mimic":
        exams["file_name"] = exams.parallel_apply(
            lambda row: str(row["file_name"]).replace(
                "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/",
                "",
            ),
            axis=1,
        )
        exams["valid"] = exams.parallel_apply(
            lambda row: check_sample(
                os.path.join(data_folder, str(row["file_name"])),
                filter_10seconds,
                check_variance,
            ),
            axis=1,
        )

    elif dataset == "ptbxl":
        exams["valid"] = exams.parallel_apply(
            lambda row: check_sample(
                os.path.join(data_folder, str(row["filename_hr"])),
                filter_10seconds,
                check_variance,
            ),
            axis=1,
        )

    elif dataset in ["chapman", "code", "heedb", "code15"]:
        exams["valid"] = exams.parallel_apply(
            lambda row: check_sample(
                os.path.join(data_folder, str(row["file_name"])),
                filter_10seconds,
                check_variance
            ),
            axis=1,
        )

    # --------------------
    # Filter + save
    # --------------------
    to_remove = exams[~exams["valid"]]
    typer.echo(f"Removed samples: {len(to_remove)}")

    exams = exams[exams["valid"]].drop(columns=["valid"])
    typer.echo(f"Final rows: {len(exams)}")

    output_path = os.path.join(
        data_folder, f"{label_file_output}.csv"
    )
    exams.to_csv(output_path, index=False)
    typer.echo(f"Saved to {output_path}")


if __name__ == "__main__":
    app()