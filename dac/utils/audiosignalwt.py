import copy
import typing
import warnings
from collections import namedtuple
from pathlib import Path

import pywt
import ptwt
import numpy as np
import torch
from audiotools import AudioSignal, STFTParams
from audiotools.ml import BaseModel

DWTParams = namedtuple(
    "DWTParams",
    ["wavelet_name"],
)

CWTParams = namedtuple(
    "CWTParams",
    ["wavelet_name", "scales"],
)

BaseModel.EXTERN.extend(["pywt", "ptwt"])


class AudioSignalDWT(AudioSignal):
    def __init__(
            self,
            audio_path_or_array: typing.Union[torch.Tensor, str, Path, np.ndarray],
            sample_rate: int = None,
            stft_params: STFTParams = None,
            dwt_params: DWTParams = None,
            offset: float = 0,
            duration: float = None,
            device: str = None,
    ):
        if dwt_params:
            self.dwt_params = dwt_params
        else:
            self.dwt_params = DWTParams(wavelet_name="db4")

        self.dwt_data = None

        super().__init__(audio_path_or_array, sample_rate, stft_params, offset, duration, device)

    def dwt(self, wavelet_name: typing.Optional[str] = None):
        wavelet_name = (
            self.dwt_params.wavelet_name if wavelet_name is None else wavelet_name
        )

        wavelet = pywt.Wavelet(wavelet_name)

        audio_data = self.audio_data
        dwt_data = ptwt.wavedec(audio_data.reshape(-1, audio_data.shape[-1]), wavelet)

        return dwt_data

    def clone(self):
        """Clones all tensors contained in the AudioSignal,
        and returns a copy of the signal with everything
        cloned. Useful when using AudioSignal within autograd
        computation graphs.

        Relevant attributes are the stft data, the audio data,
        and the loudness of the file.

        Returns
        -------
        AudioSignal
            Clone of AudioSignal.
        """
        clone = type(self)(
            self.audio_data.clone(),
            self.sample_rate,
            stft_params=self.stft_params,
            dwt_params=self.dwt_params
        )
        if self.stft_data is not None:
            clone.stft_data = self.stft_data.clone()
        if self._loudness is not None:
            clone._loudness = self._loudness.clone()
        if self.dwt_data is not None:
            clone.dwt_data = self.dwt_data.clone()
        clone.path_to_file = copy.deepcopy(self.path_to_file)
        clone.metadata = copy.deepcopy(self.metadata)
        return clone

    def detach(self):
        """Detaches tensors contained in AudioSignal.

        Relevant attributes are the stft data, the audio data,
        and the loudness of the file.

        Returns
        -------
        AudioSignal
            Same signal, but with all tensors detached.
        """
        if self._loudness is not None:
            self._loudness = self._loudness.detach()
        if self.stft_data is not None:
            self.stft_data = self.stft_data.detach()
        if self.dwt_data is not None:
            self.dwt_data = self.dwt_data.detach()

        self.audio_data = self.audio_data.detach()
        return self

    # Tensor operations
    def to(self, device: str):
        """Moves all tensors contained in signal to the specified device.

        Parameters
        ----------
        device : str
            Device to move AudioSignal onto. Typical values are
            "cuda", "cpu", or "cuda:n" to specify the nth gpu.

        Returns
        -------
        AudioSignal
            AudioSignal with all tensors moved to specified device.
        """
        if self._loudness is not None:
            self._loudness = self._loudness.to(device)
        if self.stft_data is not None:
            self.stft_data = self.stft_data.to(device)
        if self.dwt_data is not None:
            self.dwt_data = self.dwt_data.to(device)
        if self.audio_data is not None:
            self.audio_data = self.audio_data.to(device)
        return self

    @property
    def device(self):
        """Get device that AudioSignal is on.

        Returns
        -------
        torch.device
            Device that AudioSignal is on.
        """
        if self.audio_data is not None:
            device = self.audio_data.device
        elif self.stft_data is not None:
            device = self.stft_data.device
        elif self.dwt_data is not None:
            device = self.dwt_data.device
        return device

    @property
    def dwt_data(self):
        """Returns the DWT data inside the signal. Shape is
        (batch, channels, frequencies, time).

        Returns
        -------
        torch.Tensor
            Complex spectrogram data.
        """
        return self._dwt_data

    @dwt_data.setter
    def dwt_data(self, data: typing.Union[torch.Tensor, np.ndarray]):
        if data is not None:
            assert torch.is_tensor(data) and torch.is_complex(data)
            if self.dwt_data is not None and self.dwt_data.shape != data.shape:
                warnings.warn("dwt_data changed shape")
        self._dwt_data = data
        return

    # Representation
    def _info(self):
        info = super()._info()
        info["dwt_params"] = self.dwt_params

        return info

    # Indexing
    def __getitem__(self, key):
        if torch.is_tensor(key) and key.ndim == 0 and key.item() is True:
            assert self.batch_size == 1
            audio_data = self.audio_data
            _loudness = self._loudness
            stft_data = self.stft_data
            dwt_data = self.dwt_data

        elif isinstance(key, (bool, int, list, slice, tuple)) or (
                torch.is_tensor(key) and key.ndim <= 1
        ):
            # Indexing only on the batch dimension.
            # Then let's copy over relevant stuff.
            # Future work: make this work for time-indexing
            # as well, using the hop length.
            audio_data = self.audio_data[key]
            _loudness = self._loudness[key] if self._loudness is not None else None
            stft_data = self.stft_data[key] if self.stft_data is not None else None
            dwt_data = self.dwt_data[key] if self.dwt_data is not None else None

        sources = None

        copy = type(self)(audio_data, self.sample_rate, stft_params=self.stft_params, dwt_params=self.dwt_params)
        copy._loudness = _loudness
        copy._stft_data = stft_data
        copy._dwt_data = dwt_data
        copy.sources = sources

        return copy

    def __setitem__(self, key, value):
        if not isinstance(value, type(self)):
            self.audio_data[key] = value
            return

        if torch.is_tensor(key) and key.ndim == 0 and key.item() is True:
            assert self.batch_size == 1
            self.audio_data = value.audio_data
            self._loudness = value._loudness
            self.stft_data = value.stft_data
            self.dwt_data = value.dwt_data
            return

        elif isinstance(key, (bool, int, list, slice, tuple)) or (
            torch.is_tensor(key) and key.ndim <= 1
        ):
            if self.audio_data is not None and value.audio_data is not None:
                self.audio_data[key] = value.audio_data
            if self._loudness is not None and value._loudness is not None:
                self._loudness[key] = value._loudness
            if self.stft_data is not None and value.stft_data is not None:
                self.stft_data[key] = value.stft_data
            if self.dwt_data is not None and value.dwt_data is not None:
                self.dwt_data[key] = value.dwt_data
            return
