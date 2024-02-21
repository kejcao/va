import transcribe
import gui

import requests
import replicate
import time
import json
import threading
from llama_cpp import Llama

METHOD = 'API'  # either API or LOCAL

(RUNNING, STOPED, REQUEST_STOP) = range(3)
llm_context = """\
Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately. Bob always responds in less than 32 words.

User: Hello, Bob.
Bob: Hello. How may I help you today?
User: Please tell me the largest city in Europe.
Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
"""
llm_running = STOPED

llm = None
if METHOD == 'LOCAL':
    llm = Llama(model_path='./models/gemma-7b-it-Q4_K_M.gguf', seed=67)


def stream(prompt):
    if METHOD == 'API':
        for token in replicate.stream(
            'meta/llama-2-7b',
            input={'prompt': prompt, 'temperature': 0.5, 'stop_sequences': ['User:']},
        ):
            yield str(token)
    elif METHOD == 'LOCAL':
        for d in llm(prompt, max_tokens=32, stop=['User:'], echo=False, stream=True):
            yield d['choices'][0]['text']


def prompt_llm(prompt, quiet=False):
    global llm_context, llm_running

    print('BOT | ', end='', flush=True)

    llm_context += f'User: {prompt}\n'
    reply = ''
    start = time.time()

    for token in stream(llm_context):
        if not token.strip():
            continue

        if llm_running == REQUEST_STOP:
            break

        reply += token
        print(token, flush=True, end='')
        if reply[:7].lower().strip() == 'goodbye':
            break

        # polling sort of bad :/ maybe killable threads?
        while time.time() - start < 0.1:
            time.sleep(0.01)
            if llm_running == REQUEST_STOP:
                break
        start = time.time()

    # preamble, to cleanly exit
    print()
    print()
    transcribe.reset_stt()
    llm_context += reply.strip() + '\n'
    llm_running = STOPED


def callback(transcribed_text, activity):
    global llm_running

    def update_gui(**kwargs):
        gui.print_user_input(f'{activity[-1]:.1f}> ', transcribed_text, **kwargs)

    # so while the chatbot was responding, the user interrupted
    if llm_running == RUNNING and transcribed_text:
        llm_running = REQUEST_STOP
        return

    if llm_running == STOPED:
        update_gui()
        if (
            transcribed_text
            and len(activity) > 2
            and all(x < 0.2 for x in activity[-2:])
        ):
            llm_running = RUNNING
            update_gui(erase=False, grayout=True)
            threading.Thread(
                target=prompt_llm, args=(transcribed_text,), daemon=True
            ).start()
            transcribe.reset_stt()


transcribe.start(callback)
