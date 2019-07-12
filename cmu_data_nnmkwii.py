from nnmnkwii.preprocessing import trim_zeros_frames
from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii.datasets import cmu_arctic
import pysptk
import pyworld

class MyFileDataSource(cmu_arctic.WavFileDataSource):
    def __init__(self, data_root, speakers, max_files=100):
        super(MyFileDataSource, self).__init__(
            data_root, speakers, max_files=100)

    def collect_features(self, path):
        """Compute mel-cepstrum given a wav file."""
        fs, x = wavfile.read(path)
        x = x.astype(np.float64)
        f0, timeaxis = pyworld.dio(x, fs, frame_period=5)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        spectrogram = trim_zeros_frames(spectrogram)
        mc = pysptk.sp2mc(spectrogram, order=24, alpha=0.41)
        return mc.astype(np.float32)

DATA_ROOT = "/home/ryuichi/data/cmu_arctic/" # your data path
data_source = MyFileDataSource(DATA_DIR, speakers=["clb"], max_files=100)

# 100 wav files of `clb` speaker will be collected
X = FileSourceDataset(data_source)
assert len(X) == 100

for x in X:
    # do anything on acoustic features (e.g., save to disk)
    pass
