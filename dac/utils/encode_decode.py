import math
import warnings
from pathlib import Path

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core import util
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error

from dac.utils import load_model
from scripts.dataloader import MyDataset

warnings.filterwarnings("ignore", category=UserWarning)


@argbind.bind(group="encode_decode", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def encode_decode(
    input: str,
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    n_quantizers: int = None,
    device: str = "cuda",
    model_type: str = "44khz",
    win_duration: float = 5.0,
    verbose: bool = False,
):
    """Encode audio files in input path to .dac format.

    Parameters
    ----------
    input : str
        Path to input audio file or directory
    output : str, optional
        Path to output directory, by default "". If `input` is a directory, the directory sub-tree relative to `input` is re-created in `output`.
    weights_path : str, optional
        Path to weights file, by default "". If not specified, the weights file will be downloaded from the internet using the
        model_tag and model_type.
    model_tag : str, optional
        Tag of the model to use, by default "latest". Ignored if `weights_path` is specified.
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
    n_quantizers : int, optional
        Number of quantizers to use, by default None. If not specified, all the quantizers will be used and the model will compress at maximum bitrate.
    device : str, optional
        Device to use, by default "cuda"
    model_type : str, optional
        The type of model to use. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz". Ignored if `weights_path` is specified.
    """
    generator = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=weights_path,
    )
    generator.to(device)
    generator.eval()
    kwargs = {"n_quantizers": n_quantizers}

    # Find all audio files in input path
    # input = Path(input)
    # audio_files = util.find_audio(input)
    data_file = torch.load(input)
    dataset = MyDataset(data_file)

    errors = []

    for i in tqdm(range(len(dataset)), desc="Encoding files"):
        # Load file
        signal = dataset[i]["signal"]

        # Encode audio to .dac format
        encoded = generator.compress(signal, win_duration, verbose=verbose, **kwargs)
        decoded = generator.decompress(encoded, verbose=verbose)

        true = signal.numpy().squeeze()
        predicted = decoded.numpy().squeeze()

        mape = mean_absolute_percentage_error(true, predicted)
        errors.append(mape)

    print(errors)
    print(f"Mean mape: {np.mean(errors)}, median mape: {np.median(errors)}, std dev mape: {np.std(errors)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode_decode()
