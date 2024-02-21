# Voice Assistant

My attempt to make a voice assistant. Inspired by [LAION's BUD-E](https://github.com/LAION-AI/natural_voice_assistant/), but I couldn't get their code to work on my CPU.

It uses [Nvidia](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_80ms)'s STT En FastConformer Hybrid Transducer-CTC Large Streaming 80ms for local STT. There are two options for the LLM backend:

- Use the Replicate API. In which case, make sure you export your API key as an environmental variable. [More details here](https://replicate.com/docs/get-started/python).
- Use the Python bindings for `llama.cpp` to run LLM weights in the `models` directory locally—I use Google's Gemma-7b, but you could use Mixtral if you have the hardware or something else like Microsoft's Phi.

## To run

Make sure you setup one of the LLM backends. To actually activate the voice assistant, install the dependences and run `main.py`:

```
$ pip install -r requirements.txt
$ python3 main.py
```
