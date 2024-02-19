# import nemo.collections.asr as nemo_asr
#
# asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('nvidia/stt_en_conformer_ctc_small')
# print(asr_model.transcribe(['2086-149220-0033.wav']))

import copy
import time
import pyaudio
import numpy as np
import torch

from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    'stt_en_fastconformer_hybrid_large_streaming_80ms'
)
asr_model.change_decoding_strategy(decoder_type='ctc')

# make sure the model's decoding strategy is optimal
decoding_cfg = asr_model.cfg.decoding
with open_dict(decoding_cfg):
    # save time by doing greedy decoding and not trying to record the alignments
    decoding_cfg.strategy = 'greedy'
    decoding_cfg.preserve_alignments = False
    if hasattr(asr_model, 'joint'):  # if an RNNT model
        # restrict max_symbols to make sure not stuck in infinite loop
        decoding_cfg.greedy.max_symbols = 10
        # sensible default parameter, but not necessary since batch size is 1
        decoding_cfg.fused_batch_size = -1
    asr_model.change_decoding_strategy(decoding_cfg)

# set model to eval mode
asr_model.eval()

# get parameters to use as the initial cache state
(
    cache_last_channel,
    cache_last_time,
    cache_last_channel_len,
) = asr_model.encoder.get_initial_cache_state(batch_size=1)


# init params we will use for streaming
previous_hypotheses = None
pred_out_stream = None
pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
# cache-aware models require some small section of the previous processed_signal to
# be fed in at each timestep - we initialize this to a tensor filled with zeros
# so that we will do zero-padding for the very first chunk(s)
num_channels = asr_model.cfg.preprocessor.features
cache_pre_encode = torch.zeros(
    (1, num_channels, pre_encode_cache_size), device=asr_model.device
)


def reset():
    global cache_last_channel, cache_last_time, cache_last_channel_len
    global previous_hypotheses, pred_out_stream
    global cache_pre_encode

    (
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
    ) = asr_model.encoder.get_initial_cache_state(batch_size=1)

    previous_hypotheses = None
    pred_out_stream = None
    pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]

    num_channels = asr_model.cfg.preprocessor.features
    cache_pre_encode = torch.zeros(
        (1, num_channels, pre_encode_cache_size), device=asr_model.device
    )


# helper function for extracting transcriptions
def extract_transcriptions(hyps):
    """
    The transcribed_texts returned by CTC and RNNT models are different.
    This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions


# define functions to init audio preprocessor and to
# preprocess the audio (ie obtain the mel-spectrogram)
def init_preprocessor(asr_model):
    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    cfg.preprocessor.normalize = 'None'

    preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
    preprocessor.to(asr_model.device)

    return preprocessor


preprocessor = init_preprocessor(asr_model)


def preprocess_audio(audio, asr_model):
    device = asr_model.device

    # doing audio preprocessing
    audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
    audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
    processed_signal, processed_signal_length = preprocessor(
        input_signal=audio_signal, length=audio_signal_len
    )
    return processed_signal, processed_signal_length


def transcribe_chunk(new_chunk):
    global cache_last_channel, cache_last_time, cache_last_channel_len
    global previous_hypotheses, pred_out_stream
    global cache_pre_encode

    # new_chunk is provided as np.int16, so we convert it to np.float32
    # as that is what our ASR models expect
    audio_data = new_chunk.astype(np.float32) / 32768

    # get mel-spectrogram signal & length
    processed_signal, processed_signal_length = preprocess_audio(audio_data, asr_model)

    # prepend with cache_pre_encode
    processed_signal = torch.cat([cache_pre_encode, processed_signal], dim=-1)
    processed_signal_length += cache_pre_encode.shape[1]

    # save cache for next time
    cache_pre_encode = processed_signal[:, :, -pre_encode_cache_size:]

    with torch.no_grad():
        (
            pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = asr_model.conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=pred_out_stream,
            drop_extra_pre_encoded=None,
            return_transcription=True,
        )

    final_streaming_tran = extract_transcriptions(transcribed_texts)

    return final_streaming_tran[0]


# ---

chunk_size = 80 + 80  # chunk size ms + encoder step length ms

# open the microphone

p = pyaudio.PyAudio()

import requests
import time
import json
import os
import threading

# def clear_row():
#     cols, rows = os.get_terminal_size()
#     print(' ' * cols, end='\r')

prev = ('', time.time())
llm_context = None
state = False


def callback(in_data, frame_count, time_info, status):
    global llm_context

    # if state:
    #
    text = transcribe_chunk(np.frombuffer(in_data, dtype=np.int16))

    def ask(prompt):
        global llm_context

        print('BOT |', end='')
        with requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'phi',
                'prompt': prompt,
                'stream': True,
                'context': llm_context,
            },
            stream=True,
        ) as resp:
            last_line = {}
            for line in resp.iter_lines():
                print((last_line := json.loads(line))['response'], flush=True, end='')

        llm_context = last_line['context']

    if text and text.split()[-1] == 'please':
        state = True

        prompt = ' '.join(text.split()[:-1])
        print(f'> {prompt}')

        ask(prompt)

        text = ''
        reset()

        state = False

    print(f'> {text}\r', end='')

    return (in_data, pyaudio.paContinue)


SAMPLE_RATE = 16000
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    # input_device_index=6, # choose input_device_index if you have to
    stream_callback=callback,
    frames_per_buffer=int(SAMPLE_RATE * chunk_size / 1000) - 1,
)

# finally start the main loop, after all that initialization

print()
print('Listening...')

stream.start_stream()
try:
    while stream.is_active():
        time.sleep(0.1)
except:
    print()
    stream.stop_stream()
    stream.close()
    p.terminate()
