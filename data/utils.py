import math
import tempfile
import warnings

import ffmpeg

# import menpo
import scipy.io.wavfile as wav
import torch

# from parallel_wavegan.bin.preprocess import logmelfilterbank
# from parallel_wavegan.utils.utils import read_hdf5
from sklearn.preprocessing import StandardScaler

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    import torchaudio

    torchaudio.set_audio_backend("sox_io")


def to_log_mel_spec(melspec, multiplier=10, amin=1e-10, ref=torch.max, top_db=80, normalize=True):
    ref_value = math.log10(max(amin, ref(melspec).item()))
    logmelspec = torchaudio.functional.amplitude_to_DB(melspec, multiplier, amin, ref_value, top_db)
    if normalize:
        logmelspec /= 80.0
        logmelspec = (logmelspec + 0.5) / 0.5
    return logmelspec


def get_melspec(audio, spf, frames_per_clip, sample_rate, win_length, audio2video, n_fft, n_mels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=int(round(spf / audio2video)),
            n_fft=n_fft,
            n_mels=n_mels,
        )(audio)
    # number of audio frames may be 1 or 2 larger than expected due to rounding errors
    if melspec.shape[1] > frames_per_clip * audio2video:
        dif_audio_frames = melspec.shape[1] - frames_per_clip * audio2video
        melspec = melspec[:, :-dif_audio_frames]
    assert melspec.shape[1] == frames_per_clip * audio2video

    logmelspec = to_log_mel_spec(melspec)
    logmelspec = logmelspec.transpose(0, 1)
    logmelspec = logmelspec.unsqueeze(0)

    return logmelspec


# Code mostly inherited from bulletin
# Save video sample in train dir for debugging
def save_av_sample(video, video_rate, audio=None, audio_rate=16_000, path=None):
    video_save = 0.165 * video.numpy() + 0.421
    temp_filename = next(tempfile._get_candidate_names())
    if path:
        video_path = path
    else:
        video_path = "/tmp/" + next(tempfile._get_candidate_names()) + ".mp4"
    menpo.io.export_video(
        [menpo.image.Image(frame, copy=False) for frame in video_save],
        "/tmp/" + temp_filename + ".mp4",
        fps=video_rate,
        overwrite=True,
    )
    audio_save = audio.squeeze().numpy()
    wav.write("/tmp/" + temp_filename + ".wav", audio_rate, audio_save)
    in1 = ffmpeg.input("/tmp/" + temp_filename + ".mp4")
    in2 = ffmpeg.input("/tmp/" + temp_filename + ".wav")
    out = ffmpeg.output(in1["v"], in2["a"], video_path, loglevel="panic").overwrite_output()
    try:
        out.run(quiet=True)
        print("Saved AV Sample!")
    except:
        pass


def rodrigo_melspec(x, sr=16_000, fft_size=2048, hop_size=300, win_length=1200, num_mels=80, fmin=80, fmax=7600):
    scaler = StandardScaler()
    scaler.mean_ = read_hdf5("/vol/paramonos2/projects/rodrigo/video2audio3/conf/model/spec2wav/stats.h5", "mean")
    scaler.scale_ = read_hdf5("/vol/paramonos2/projects/rodrigo/video2audio3/conf/model/spec2wav/stats.h5", "scale")
    scaler.n_features_in_ = scaler.mean_.shape[0]
    mel = logmelfilterbank(
        x, sr, fft_size=fft_size, hop_size=hop_size, win_length=win_length, num_mels=num_mels, fmin=fmin, fmax=fmax
    )
    mel = scaler.transform(mel)
    return mel
