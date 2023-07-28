import librosa
import json
import numpy as np
import tensorflow as tf
import math
import statistics as st
from enum import Enum
import genres

INSUFFICIENT_DATA = -1
WRONG_FORMAT = -2

def input_audio_preprocessing (input_file, num_mfcc=20, num_fft=2048, hop_length=512):
    try:
        signal, sr = librosa.load(input_file)
    except:
        return WRONG_FORMAT
    duration = librosa.get_duration(y=signal, sr=sr)
    if duration < 30:
        return INSUFFICIENT_DATA
    samples_in_track = duration * sr
    num_samples_per_segment = 132300
    expected_num_mfcc_vectors_per_segment = 259
    num_seg = math.ceil(samples_in_track / num_samples_per_segment)

    input_array = []
    for s in range(num_seg):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_mfcc=num_mfcc,
                                                n_fft=num_fft,
                                                hop_length = hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        input_array.append(mfcc)
    return np.array(input_array)


def predict_from_input(model, input_array):
    prediction = model.predict(input_array)
    arr_pred = []

    for arr in prediction:
        arr_pred.append(np.argmax(arr, axis=0))

    return st.mode(arr_pred)
