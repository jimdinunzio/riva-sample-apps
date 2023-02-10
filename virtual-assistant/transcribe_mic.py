# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import time
import riva.client
from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters
import server_api as serv_api

import riva.client.audio_io


def parse_args() -> argparse.Namespace:
    default_device_info = riva.client.audio_io.get_default_input_device_info()
    default_device_index = None if default_device_info is None else default_device_info['index']
    parser = argparse.ArgumentParser(
        description="Streaming transcription from microphone via Riva AI Services",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-device", type=int, default=default_device_index, help="An input audio device to use.")
    parser.add_argument("--list-devices", action="store_true", help="List input audio device indices.")
    parser = add_asr_config_argparse_parameters(parser, profanity_filter=True)
    parser = add_connection_argparse_parameters(parser)
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        help="A number of frames per second in audio streamed from a microphone.",
        default=16000,
    )
    parser.add_argument(
        "--file-streaming-chunk",
        type=int,
        default=1600,
        help="A maximum number of frames in a audio chunk sent to server.",
    )
    args = parser.parse_args()
    return args

def speak(tts_service, sound_stream, out_f, text):
    print("Generating audio for request...")
    print(f"  > '{text}': ", end='')
    start = time.time()
    resp = tts_service.synthesize(text, 'English-US.Male-1', 'en-US', sample_rate_hz=44100)
    stop = time.time()
    print(f"Time spent: {(stop - start):.3f}s")
    if sound_stream is not None:
        sound_stream(resp.audio)
    if out_f is not None:
        out_f.writeframesraw(resp.audio)

def main() -> None:
    args = parse_args()
    if args.list_devices:
        riva.client.audio_io.list_input_devices()
        return
    nchannels = 1
    sampwidth = 2
    sound_stream, out_f = None, None        
    auth = riva.client.Auth(args.ssl_cert, args.use_ssl, args.server)
    tts_service = riva.client.SpeechSynthesisService(auth)

    try:
        sound_stream = riva.client.audio_io.SoundCallBack(
            output_device_index=None, nchannels=nchannels, sampwidth=sampwidth, framerate=44100
        )

        asr_service = riva.client.ASRService(auth)
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=args.language_code,
                max_alternatives=1,
                profanity_filter=args.profanity_filter,
                enable_automatic_punctuation=True,
                verbatim_transcripts=not args.no_verbatim_transcripts,
                sample_rate_hertz=args.sample_rate_hz,
                audio_channel_count=1,
            ),
            interim_results=True,
        )
        riva.client.add_word_boosting_to_config(config, args.boosted_lm_words, args.boosted_lm_score)
        context = {}
        resp, context, index, debug = serv_api.get_input("", {}, "rivaWeather", None, 1)
        resp_text = resp[0]['payload']['text']
        speak(tts_service, sound_stream, None, resp_text)
        while True:            
            transcript = None
            with riva.client.audio_io.MicrophoneStream(
                args.sample_rate_hz,
                args.file_streaming_chunk,
                device=args.input_device,
            ) as audio_chunk_iterator:
                responses=asr_service.streaming_response_generator(
                    audio_chunks=audio_chunk_iterator,
                    streaming_config=config,
                    )
                speech_done = False
                while True:
                    response = next(responses) # process next chunk of audio
                    if not response.results:
                        continue
                    for result in response.results:
                        if not result.alternatives:
                            continue
                        if result.is_final:
                            transcript = result.alternatives[0].transcript
                            speech_done = True
                            break
                    if speech_done:
                        break
            
            resp, context, index, debug = serv_api.get_input(transcript, context, "rivaWeather", None, 1)
            end_interaction = False
            if resp != "Error":
                for r in resp:
                    resp_text = r['payload']['text']
                    #print(resp_text)
                    speak(tts_service, sound_stream, None, resp_text)
                    if context['intent'] == 'smalltalk.personality_goodbye':
                        end_interaction = True
                        break
                if end_interaction:
                    break
    finally:
        if out_f is not None:
            out_f.close()
        if sound_stream is not None:
            sound_stream.close()

if __name__ == '__main__':
    main()
