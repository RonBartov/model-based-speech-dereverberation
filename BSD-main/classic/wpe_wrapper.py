import numpy as np
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft


class ClassicWPE:
    def __init__(self, taps=10, delay=3, iterations=5, stft_size=512, stft_shift=128):
        self.taps = taps
        self.delay = delay
        self.iterations = iterations
        self.stft_options = dict(size=stft_size, shift=stft_shift)

    def dereverb_batch(self, z_batch):
        """
        Apply WPE to a batch of multichannel reverberant signals.

        Parameters:
        - z_batch: np.ndarray of shape (batch, samples, channels)

        Returns:
        - np.ndarray of shape (batch, samples), mono dereverberated signals
        """
        output = []
        for z in z_batch:  # z: (samples, channels)
            y = self._apply_wpe(z)  # (samples, channels)
            y_mono = np.sum(y, axis=1)  # mono signal (samples,)
            output.append(y_mono)
        return np.stack(output)  # (batch, samples)

    def _apply_wpe(self, z):
        """
        Apply WPE to a single multichannel signal.

        Parameters:
        - z: np.ndarray of shape (samples, channels)

        Returns:
        - np.ndarray of shape (samples, channels)
        """
        y = z.T  # (channels, samples)
        Y = stft(y, **self.stft_options).transpose(2, 0, 1)  # (frames, freq, channels)
        Z = wpe(Y, taps=self.taps, delay=self.delay, iterations=self.iterations).transpose(1, 2, 0)  # (channels, frames, freq)
        z_hat = istft(Z, size=self.stft_options['size'], shift=self.stft_options['shift']).T  # (samples, channels)
        return z_hat