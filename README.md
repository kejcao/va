# Voice Assistant

My attempt to make a voice assistant. Inspired by [LAION](https://github.com/LAION-AI/natural_voice_assistant/), but I couldn't get their code to work on my CPU.

Code assumes you have [ollama](https://github.com/ollama/ollama) running in the background. To the voice assistant run:

```
$ pip install -r requirements.txt
$ python3 main.py
```

Internally, it uses:

- [Nvidia](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_80ms)'s STT En FastConformer Hybrid Transducer-CTC Large Streaming 80ms for STT.
- Ollama to run the LLM.
