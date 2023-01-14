# ==============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# The License information can be found under the "License" section of the
# README.md file.
# ==============================================================================

from __future__ import division

import uuid
import time
import logging
from os.path import dirname, abspath, join, isdir
from os import listdir
from config import client_config
from engineio.payload import Payload

from riva.chatbot.chatbots_multiconversations_management import create_chatbot, get_new_user_conversation_index, get_chatbot
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
Payload.max_decode_packets = 500  # https://github.com/miguelgrinberg/python-engineio/issues/142
verbose = client_config['VERBOSE']


def get_newuser_conversation_index():
    return get_new_user_conversation_index()

# Audio source for TTS
def audio(user_conversation_index, post_id):
    if verbose:
        print(f'[{user_conversation_index}] audio speak: {post_id}')
    currentChatbot = get_chatbot(user_conversation_index)
    return currentChatbot.get_tts_speech()

# Handles ASR audio transcript output
def stream(user_conversation_index):
    def audio_stream():
        currentChatbot = get_chatbot(user_conversation_index)
        if currentChatbot:
            asr_transcript = currentChatbot.get_asr_transcript()
            for t in asr_transcript:
                yield t
        params = {'response': "Audio Works"}
        return params
    return audio_stream()


# Used for sending messages to the bot
def get_input(text, context, bot, payload, user_conversation_index):
    if user_conversation_index:
        create_chatbot(user_conversation_index, None, verbose=client_config['VERBOSE'])
        currentChatBot = get_chatbot(user_conversation_index)
        try:
            response = currentChatBot.stateDM.execute_state(
                bot, context, text)

            if client_config['DEBUG']:
                print(f"[{user_conversation_index}] Response from RivaDM: {response}")

            for resp in response['response']:
                speak = resp['payload']['text']
                if len(speak):
                    currentChatBot.tts_fill_buffer(speak)
            return response['response'], response['context'], user_conversation_index, client_config["DEBUG"]
        except Exception as e:  # Error in execution

            print(e)
            return "Error during execution."
    else:
        print("user_conversation_index not found")
        return "user_conversation_index not found"


# Writes audio data to ASR buffer
def receive_remote_audio(data):
    currentChatbot = get_chatbot(data["user_conversation_index"])
    if currentChatbot:
        currentChatbot.asr_fill_buffer(data["audio"])


def start_tts(data):
    currentChatbot = get_chatbot(data["user_conversation_index"])
    if currentChatbot:
        currentChatbot.start_tts()


def stop_tts(data):
    currentChatbot = get_chatbot(data["user_conversation_index"])
    if currentChatbot:
        currentChatbot.stop_tts()


def pauseASR(data):
    currentChatbot = get_chatbot(data["user_conversation_index"])
    if currentChatbot:
        if verbose:
            print(f"[{data['user_conversation_index']}] Pausing ASR requests.")
        currentChatbot.pause_asr()


def unpauseASR(data):
    currentChatbot = get_chatbot(data["user_conversation_index"])
    if currentChatbot:
        if verbose:
            print(f"[{data['user_conversation_index']}] Attempt at Unpausing ASR requests on {data['on']}.")
        unpause_asr_successful_flag = currentChatbot.unpause_asr(data["on"])
        return unpause_asr_successful_flag

def pause_wait_unpause_asr(data):
    currentChatbot = get_chatbot(data["user_conversation_index"])
    if currentChatbot:
        currentChatbot.pause_wait_unpause_asr()
        return True

def connect():
    if verbose:
        print('[Riva Chatbot] Client connected')


def disconnect():
    if verbose:
        print('[Riva Chatbot] Client disconnected')
