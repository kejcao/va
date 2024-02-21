import transcribe
import gui

import requests
import replicate
import time
import json
import threading

llm_context = ''
llm_running = False


def prompt_llm(prompt, quiet=False):
    global llm_context, llm_running

    prompt = f'[INST] {prompt} [/INST]\n'
    llm_context += prompt

    print('BOT | ', end='', flush=True)

    reply = ''
    for event in replicate.stream(
        'meta/llama-2-7b-chat',
        input={
            'prompt': llm_context,
            # 'system_prompt': 'You are an AI assistant. Always respond as concisely as possible. Neglect to mention any unnecessary information not pertaining to the user input. Use as little words as possible.',
            'system_prompt': 'You are an AI assistant. Use as little words as possible. Your responses should not exceed 30 words.',
            'temperature': 0.1,
        },
    ):
        tok = str(event)
        reply += tok
        if tok.strip():
            print(tok, flush=True, end='')

    print()
    print()

    llm_context += reply.strip() + '\n'
    llm_running = False
    transcribe.reset_stt()


def callback(transcribed_text, activity):
    global llm_running

    def update_gui(**kwargs):
        gui.print_user_input(f'{activity[-1]:.1f}> ', transcribed_text, **kwargs)


    if transcribed_text and len(activity) > 2 and all(x < 0.2 for x in activity[-2:]):
        if not llm_running and transcribed_text:
            llm_running = True
            update_gui(erase=False, grayout=True)
            threading.Thread(
                target=prompt_llm, args=(transcribed_text,), daemon=True
            ).start()
            transcribe.reset_stt()

        # transcribe.reset_stt()

    if not llm_running:
        update_gui()


transcribe.start(callback)
