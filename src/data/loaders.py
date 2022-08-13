import pandas as pd


class DataLoader:
    @classmethod
    def from_csv(cls, file_path: str, sep: str = ",", line_terminator=None) -> pd.DataFrame:
        return pd.read_csv(file_path, sep=sep, lineterminator=line_terminator)
