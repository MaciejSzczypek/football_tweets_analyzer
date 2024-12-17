from pathlib import Path

def make_directory_if_not_exists(directory_path: str):
    directory = Path(directory_path)
    directory.mkdir(parents=True, exist_ok=True)
