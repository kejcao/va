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
llm_context = ''
if METHOD == 'LOCAL':
    llm_context = """\
    Transcript of a dialog, where User interacts with an Assistant named Bob. Bob always responds in less than 32 words.

    User: hello, bob
    Bob: Hi!
    User: please tell me the largest city in europe
    Bob: Moscow, the capital of Russia, is the largest city in Europe.
    """
llm_running = STOPED

llm = None
if METHOD == 'LOCAL':
    llm = Llama(model_path='./models/gemma-7b-it-Q4_K_M.gguf', seed=67)


def stream(prompt):
    if METHOD == 'API':
        for token in replicate.stream(
            'meta/llama-2-7b-chat',
            input={
                'prompt': prompt,
                'system_prompt': 'You are an AI assistant named Bob. Bob always adheres to the user\'s requests but responds in less than 32 words.',
                'temperature': 0.8,
            },
            # input={'prompt': prompt, 'temperature': 0.5, 'stop_sequences': '<User:>'}
        ):
            token = str(token)

            if not token.strip():
                continue

            l = token.split(' ')
            for t in l[:-1]:
                yield str(t + ' ')
            yield l[-1] + (' ' if token[-1] == ' ' else '')
    elif METHOD == 'LOCAL':
        for d in llm(prompt, max_tokens=32, stop=['User:'], echo=False, stream=True):
            yield d['choices'][0]['text']


def prompt_llm(prompt, quiet=False):
    global llm_context, llm_running

    print('Bob | ', end='', flush=True)

    llm_context += (
        f'User: {prompt}\nBob: ' if METHOD == 'local' else f'[INST]{prompt}[/INST]\n'
    )
    reply = ''
    start = time.time()

    for token in stream(llm_context):
        if llm_running == REQUEST_STOP:
            break

        reply += token
        print(token, flush=True, end='')
        if reply[:7].lower().strip() == 'goodbye':
            break

        # polling sort of bad :/ maybe killable threads?
        while time.time() - start < 0.17:
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
        gui.print_user_input('> ', transcribed_text, **kwargs)
        # gui.print_user_input(f'{activity[-1]:.1f}> ', transcribed_text, **kwargs)

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
