from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, expanduser
from torch.utils import data as data_utils
import numpy
from torch.utils.data import Dataset, DataLoader
import numpy as np

class _NPYDataSource(FileDataSource):
    def __init__(self, col):
        self.col = col
        print("self.col is", self.col)
    def collect_files(self):
        meta = join("/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Project_TAGHIM/siri_Tacotron_may2019/scripts/SGCM/LJSpeech-1.1/", "metadata.csv")
        with open(meta, "rb") as f:
            print("f is:", f)
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[0] + '.wav', lines))
###        print("lines are:", lines)
        paths = list(map(lambda f: join("/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Project_TAGHIM/siri_Tacotron_may2019/scripts/SGCM/LJSpeech-1.1/wavs/", f), lines))
        return paths

    def collect_features(self, path):
        print("loaded and returning", path)
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self):
        super(MelSpecDataSource, self).__init__(1)
        print("am in mel")

class LinearSpecDataSource(_NPYDataSource):
    def __init__(self):
        super(LinearSpecDataSource, self).__init__(0)


class PyTorchDataset(Dataset):
    def __init__(self, Mel, Y):
        print("am in datastet")
        self.Mel = Mel
        self.Y = Y

    def __getitem__(self, idx):
        print("datatste is returning", idx, self.Mel[idx], numpy.shape(self.Mel[idx]))
        return  self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.Y)


Mel = FileSourceDataset(MelSpecDataSource())
print("mel is", Mel)
Y = FileSourceDataset(LinearSpecDataSource())
print("Y i", Y)
dataset = PyTorchDataset(Mel, Y)
train_loader = DataLoader(dataset, batch_size =4, shuffle=True)
##data_loader = data_utils.DataLoader(
##        dataset, batch_size=4,
 ##       num_workers=hparams.num_workers, shuffle=True,
  ##      collate_fn=collate_fn, pin_memory=hparams.pin_memory)
for x, y in train_loader:

   print("x is:", x, "y is:", y)
