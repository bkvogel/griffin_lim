# griffin_lim
Python implementation of the Griffin and Lim algorithm to recover an audio signal from a magnitude-only spectrogram.

##### Description

This is a python implementation of Griffin and Lim's algorithm to recover an audio signal given only the magnitude of the Short-Time Fourier Transform (STFT), also known as the spectrogram. The Griffin and Lim method is described in the paper:

Griffin D. and Lim J. (1984). "Signal Estimation from Modified Short-Time Fourier Transform". IEEE Transactions on Acoustics, Speech and Signal Processing. 32 (2): 236–243. doi:10.1109/TASSP.1984.1164317

This is an iterative algorithm that attempts to find the signal having an STFT such that the magnitude part is as close as possible to the modified spectrogram (that is, magnitude STFT).

The Griffin and Lim algorithm can be useful in an audio processing system where an audio signal is transformed to a spectrogram which is then modified or in which an algorithm produces a spectrogram that we would like to "invert" back into an audio signal.

#### Requirements

Requires Python 3 (tested with Python v3.6 Anaconda distribution)

#### Usage

The provided code shows an example usage of the Griffin and Lim algorithm. It loads an audio file, computes the spectrogram, optionally performs low-pass filtering by zeroing all frequency bins above some cutoff frequency, and then uses the Griffin and Lim algorithm to reconstruct an audio signal from the modified spectrogram. Finally, both the reconstructed audio signal and the spectrogram plot figure are saved to a file.

A short audio clip, `bkvhi.wav`,  of a few piano notes is provided that can be used as the input signal. To run the example:

```
python run_demo.py
```

There are several optional command-line arguments. See `run_demo.py` for details.

For example, to enable a low-pass filter with a cutoff frequency of 1000 Hz:

```
python run_demo.py --enable_filter --cutoff_freq 1000
```

The default spectrogram uses a linear frequency scale. For some applications it can be useful to transform to the mel scale. The mel scale is a perceptually motivated frequency scale that is logarithmic in frequency. We can transform the magnitude spectrogram into a mel scale spectrogram, then approximately invert back to a linear scale spectrogram, and finally reconstruct an audio file by using the `--enable_mel_scale` option:

```
python run_demo.py --enable_mel_scale
```

Regarding the number of iterations, a small number such as 50 is good for quick runs but will probably result in some audible artifacts. For high quality output, it is suggested to run for 1000 iterations or more.

#### License

FreeBSD license.
