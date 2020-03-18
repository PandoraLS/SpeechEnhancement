# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""
import os
import nnresample
import soundfile


def resample(wav, old_sr, new_sr):
    return nnresample.resample(wav, new_sr, old_sr)


def wav_read(path, samplerate=None):
    y, sr = soundfile.read(path)
    print("语音", y)
    print("采样率", sr)
    # sr, y = wavfile.read(path)
    # y,sr = soundfile.read(io.BytesIO(urlopen(path).read()))
    if samplerate is not None and samplerate != sr:
        y = resample(y, sr, samplerate)
        sr = samplerate
    return y, sr


def wav_write(path, wav, sr):
    soundfile.write(path, wav, sr)


# def resample(wav_dir, samplerate):
#     for root, dirs, files in os.walk(wav_dir):
#         for file in files:
#             if file.endswith('wav') or file.endswith('WAV'):
#                 wav_path = os.path.join(root, file)
#                 wav, sr = wav_read(wav_path)
#                 if sr < samplerate:
#                     # 如果需要，可注释这段判断
#                     raise Exception('resample from low freq %d to high freq %d is meaningless' % (sr, samplerate))
#                 if sr != samplerate:
#                     wav_write(wav_path, wav, samplerate)


if __name__ == "__main__":
    import soundfile
    print('llllll')
    # # resample('/home2nd/pengc/workspace/audio_analysis/coolpanda/dataset/ESC-50.5', 16000)
    # path = r"C:\Users\M\Desktop\example\16_20180131_0.wav"
    # y, sr = wav_read(path)
