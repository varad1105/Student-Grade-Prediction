"""Download and validate the UCI Student Performance mathematics dataset."""

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

from src.data_processing import DATASET_URL, load_dataset

DATA_DIRECTORY = Path("data")
DATASET_PATH = DATA_DIRECTORY / "student-mat.csv"


def download_dataset() -> Path:
    """Download only student-mat.csv from the official UCI archive."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(DATASET_URL, timeout=30) as response:
            archive_bytes = response.read()
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            f"Could not download the UCI dataset: {error}\n"
            "Download student.zip manually from https://archive.ics.uci.edu/dataset/320/student+performance "
            "and place student-mat.csv in the data/ folder."
        ) from error

    try:
        from io import BytesIO

        with ZipFile(BytesIO(archive_bytes)) as archive:
            member = next(
                (name for name in archive.namelist() if Path(name).name == "student-mat.csv"),
                None,
            )
            if member is not None:
                dataset_bytes = archive.read(member)
            else:
                nested_archive = next(
                    (name for name in archive.namelist() if Path(name).name == "student.zip"),
                    None,
                )
                if nested_archive is None:
                    raise RuntimeError(
                        "The downloaded UCI archive contains neither student-mat.csv "
                        "nor the expected nested student.zip."
                    )
                with ZipFile(BytesIO(archive.read(nested_archive))) as student_archive:
                    member = next(
                        (
                            name
                            for name in student_archive.namelist()
                            if Path(name).name == "student-mat.csv"
                        ),
                        None,
                    )
                    if member is None:
                        raise RuntimeError(
                            "The nested UCI archive does not contain student-mat.csv."
                        )
                    dataset_bytes = student_archive.read(member)
            DATASET_PATH.write_bytes(dataset_bytes)
    except BadZipFile as error:
        raise RuntimeError("The UCI download was not a valid ZIP archive.") from error

    validated = load_dataset(DATASET_PATH)
    print(f"Validated {len(validated)} records and {len(validated.columns)} columns: {DATASET_PATH}")
    return DATASET_PATH


if __name__ == "__main__":
    download_dataset()
