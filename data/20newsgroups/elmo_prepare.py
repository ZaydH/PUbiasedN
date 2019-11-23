import sys
import h5py
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import nltk
import torch
from allennlp.common.file_utils import cached_path
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups

WEIGHT_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
OPTIONS_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"


def generate_dataset(ds_name):
    assert ds_name == "train" or ds_name == "test"
    newsgroups = fetch_20newsgroups(subset=ds_name, shuffle=True)

    n = len(newsgroups.data)

    def _make_elmo(n_device: int) -> ElmoEmbedder:
        return ElmoEmbedder(cached_path(OPTIONS_FILE, "."), cached_path(WEIGHT_FILE, "."), n_device)

    # First learner use CUDA, second does not
    elmos = [_make_elmo(i) for i in range(0, -2, -1)]
    data = np.zeros([n, 9216])

    if not torch.cuda.is_available():  # Has to be out of for loop or stdout overwrite messes up
        print('cuda fail')
    for i in range(n):
        A = [word_tokenize(newsgroups.data[i])]
        sys.stdout.write(f"Processing {ds_name} document {i+1}/{n}\r")
        sys.stdout.flush()
        try:
            em = elmos[0].embed_batch(A)
        except RuntimeError:
            em = elmos[1].embed_batch(A)
        em = np.concatenate(
                [np.mean(em[0], axis=1).flatten(),
                 np.min(em[0], axis=1).flatten(),
                 np.max(em[0], axis=1).flatten()])
        data[i] = em
    print("")

    f = h5py.File(f'20newsgroups_elmo_mmm_{ds_name}.hdf5', 'w')
    f.create_dataset('data', data=data)
    f.close()


if __name__ == "__main__":
    nltk.download("punkt")
    generate_dataset("train")
    generate_dataset("test")
