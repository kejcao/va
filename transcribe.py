import copy
import time
import pyaudio
import numpy as np
import torch
import signal
import sys

from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

# make the stt model
stt_model = nemo_asr.models.ASRModel.from_pretrained(
    'stt_en_fastconformer_hybrid_large_streaming_80ms'
)
stt_model.change_decoding_strategy(decoder_type='rnnt')

# make sure the model's decoding strategy is optimal
decoding_cfg = stt_model.cfg.decoding
with open_dict(decoding_cfg):
    # save time by doing greedy decoding and not trying to record the alignments
    decoding_cfg.strategy = 'greedy'
    decoding_cfg.preserve_alignments = False
    if hasattr(stt_model, 'joint'):  # if an RNNT model
        # restrict max_symbols to make sure not stuck in infinite loop
        decoding_cfg.greedy.max_symbols = 10
        # sensible default parameter, but not necessary since batch size is 1
        decoding_cfg.fused_batch_size = -1
    stt_model.change_decoding_strategy(decoding_cfg)

stt_model.eval()

# init STT constants
pre_encode_cache_size = stt_model.encoder.streaming_cfg.pre_encode_cache_size[1]
num_channels = stt_model.cfg.preprocessor.features

# init variables for streaming STT
(
    cache_last_channel,
    cache_last_time,
    cache_last_channel_len,
    previous_hypotheses,
    pred_out_stream,
    cache_pre_encode,
) = None, None, None, None, None, None


def reset_stt():
    global transcribed_text
    transcribed_text = ''

    global cache_last_channel, cache_last_time, cache_last_channel_len
    global previous_hypotheses, pred_out_stream
    global cache_pre_encode

    # get parameters to use as the initial cache state
    (
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
    ) = stt_model.encoder.get_initial_cache_state(batch_size=1)

    # init params we will use for streaming
    previous_hypotheses = None
    pred_out_stream = None

    # cache-aware models require some small section of the previous processed_signal to
    # be fed in at each timestep - we initialize this to a tensor filled with zeros
    # so that we will do zero-padding for the very first chunk(s)
    cache_pre_encode = torch.zeros(
        (1, num_channels, pre_encode_cache_size), device=stt_model.device
    )


reset_stt()


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


preprocessor = init_preprocessor(stt_model)


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
    processed_signal, processed_signal_length = preprocess_audio(audio_data, stt_model)

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
        ) = stt_model.conformer_stream_step(
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


SAMPLE_RATE = 16000
chunk_size = 80 + 80  # chunk size ms + encoder step length ms
frames_per_buffer = int(SAMPLE_RATE * chunk_size / 1000) - 1

# init VAD
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


def start(callback_):
    transcribed_text = ''
    activity = []

    def callback(in_data, frame_count, time_info, status):
        audio = np.frombuffer(in_data, dtype=np.int16)
        activity.append(
            vad_model(torch.from_numpy(int2float(audio)), SAMPLE_RATE).item()
        )

        callback_(transcribe_chunk(audio), activity)
        return (in_data, pyaudio.paContinue)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        input=True,
        channels=1,
        rate=SAMPLE_RATE,
        frames_per_buffer=frames_per_buffer,
        stream_callback=callback,
        # input_device_index=6, # if microphone audio doesn't work, try differing this
    )

    def quit():
        print()
        stream.stop_stream()
        stream.close()
        p.terminate()
        sys.exit(0)

    stream.start_stream()
    try:
        while stream.is_active():
            time.sleep(0.1)
    except:
        quit()
