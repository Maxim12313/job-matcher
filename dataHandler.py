from collections.abc import Iterator
from datasets import load_dataset
from torch.utils.data import DataLoader


def keepRow(row):
    # just keep if none of of cols are none
    return all(x is not None for x in row)


class DataHandler:
    def generate(self):
        loader = DataLoader(self.ds, batch_size=4)  # TODO: change batch_size
        for batch in loader:
            data = batch["Resume_test"]

    def __iter__(self):
        return self.generate()

    def __init__(self):
        ds = load_dataset("json", data_files=path + "entity-recognition")
        ds = ds.filter(keepRow)
        # ds = ds.take(16)  # TODO: remove
        # self.ds = ds.with_format("torch")


if __name__ == "__main__":
    data = DataHandler()
