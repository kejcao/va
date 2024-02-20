import transcribe
import gui

import requests
import time
import json
import threading

llm_context = None
llm_running = False


def prompt_llm(prompt, quiet=False):
    global llm_context, llm_running

    if not quiet:
        print('BOT | ', end='', flush=True)

    first = True
    with requests.post(
        'http://localhost:11434/api/generate',
        json={
            # 'model': 'dolphin-mistral',
            # 'model': 'starling-lm:7b-alpha',
            'model': 'phi:chat',
            'prompt': prompt,
            'stream': True,
            'context': llm_context,
            'system': 'You are an AI assistant. Respond as concisely as possible.',
        },
        stream=True,
    ) as resp:
        last_line = {}
        for line in resp.iter_lines():
            if not llm_running:  # early exit?
                if not quiet:
                    print()
                return

            tok = (last_line := json.loads(line))['response']

            if not quiet:
                print(tok.strip() if first else tok, flush=True, end='')

            first = False

    if not quiet:
        print()
        print()

    try:
        llm_context = last_line['context']
    except:
        pass

    llm_running = False


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


# ask('.', quiet=True) # warm up the large language model
transcribe.start(callback)
