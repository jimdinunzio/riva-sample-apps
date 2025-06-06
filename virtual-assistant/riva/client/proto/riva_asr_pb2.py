# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: riva/proto/riva_asr.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import riva_audio_pb2 as riva_dot_proto_dot_riva__audio__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19riva/proto/riva_asr.proto\x12\x0fnvidia.riva.asr\x1a\x1briva/proto/riva_audio.proto\"U\n\x10RecognizeRequest\x12\x32\n\x06\x63onfig\x18\x01 \x01(\x0b\x32\".nvidia.riva.asr.RecognitionConfig\x12\r\n\x05\x61udio\x18\x02 \x01(\x0c\"\x92\x01\n\x19StreamingRecognizeRequest\x12G\n\x10streaming_config\x18\x01 \x01(\x0b\x32+.nvidia.riva.asr.StreamingRecognitionConfigH\x00\x12\x17\n\raudio_content\x18\x02 \x01(\x0cH\x00\x42\x13\n\x11streaming_request\"\xba\x04\n\x11RecognitionConfig\x12,\n\x08\x65ncoding\x18\x01 \x01(\x0e\x32\x1a.nvidia.riva.AudioEncoding\x12\x19\n\x11sample_rate_hertz\x18\x02 \x01(\x05\x12\x15\n\rlanguage_code\x18\x03 \x01(\t\x12\x18\n\x10max_alternatives\x18\x04 \x01(\x05\x12\x18\n\x10profanity_filter\x18\x05 \x01(\x08\x12\x37\n\x0fspeech_contexts\x18\x06 \x03(\x0b\x32\x1e.nvidia.riva.asr.SpeechContext\x12\x1b\n\x13\x61udio_channel_count\x18\x07 \x01(\x05\x12 \n\x18\x65nable_word_time_offsets\x18\x08 \x01(\x08\x12$\n\x1c\x65nable_automatic_punctuation\x18\x0b \x01(\x08\x12/\n\'enable_separate_recognition_per_channel\x18\x0c \x01(\x08\x12\r\n\x05model\x18\r \x01(\t\x12\x1c\n\x14verbatim_transcripts\x18\x0e \x01(\x08\x12Y\n\x14\x63ustom_configuration\x18\x18 \x03(\x0b\x32;.nvidia.riva.asr.RecognitionConfig.CustomConfigurationEntry\x1a:\n\x18\x43ustomConfigurationEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"i\n\x1aStreamingRecognitionConfig\x12\x32\n\x06\x63onfig\x18\x01 \x01(\x0b\x32\".nvidia.riva.asr.RecognitionConfig\x12\x17\n\x0finterim_results\x18\x02 \x01(\x08\"/\n\rSpeechContext\x12\x0f\n\x07phrases\x18\x01 \x03(\t\x12\r\n\x05\x62oost\x18\x04 \x01(\x02\"N\n\x11RecognizeResponse\x12\x39\n\x07results\x18\x01 \x03(\x0b\x32(.nvidia.riva.asr.SpeechRecognitionResult\"\x8c\x01\n\x17SpeechRecognitionResult\x12\x43\n\x0c\x61lternatives\x18\x01 \x03(\x0b\x32-.nvidia.riva.asr.SpeechRecognitionAlternative\x12\x13\n\x0b\x63hannel_tag\x18\x02 \x01(\x05\x12\x17\n\x0f\x61udio_processed\x18\x03 \x01(\x02\"p\n\x1cSpeechRecognitionAlternative\x12\x12\n\ntranscript\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12(\n\x05words\x18\x03 \x03(\x0b\x32\x19.nvidia.riva.asr.WordInfo\"R\n\x08WordInfo\x12\x12\n\nstart_time\x18\x01 \x01(\x05\x12\x10\n\x08\x65nd_time\x18\x02 \x01(\x05\x12\x0c\n\x04word\x18\x03 \x01(\t\x12\x12\n\nconfidence\x18\x04 \x01(\x02\"Z\n\x1aStreamingRecognizeResponse\x12<\n\x07results\x18\x01 \x03(\x0b\x32+.nvidia.riva.asr.StreamingRecognitionResult\"\xb4\x01\n\x1aStreamingRecognitionResult\x12\x43\n\x0c\x61lternatives\x18\x01 \x03(\x0b\x32-.nvidia.riva.asr.SpeechRecognitionAlternative\x12\x10\n\x08is_final\x18\x02 \x01(\x08\x12\x11\n\tstability\x18\x03 \x01(\x02\x12\x13\n\x0b\x63hannel_tag\x18\x05 \x01(\x05\x12\x17\n\x0f\x61udio_processed\x18\x06 \x01(\x02\x32\xe2\x01\n\x15RivaSpeechRecognition\x12T\n\tRecognize\x12!.nvidia.riva.asr.RecognizeRequest\x1a\".nvidia.riva.asr.RecognizeResponse\"\x00\x12s\n\x12StreamingRecognize\x12*.nvidia.riva.asr.StreamingRecognizeRequest\x1a+.nvidia.riva.asr.StreamingRecognizeResponse\"\x00(\x01\x30\x01\x42\x1bZ\x16nvidia.com/riva_speech\xf8\x01\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'riva.proto.riva_asr_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z\026nvidia.com/riva_speech\370\001\001'
  _RECOGNITIONCONFIG_CUSTOMCONFIGURATIONENTRY._options = None
  _RECOGNITIONCONFIG_CUSTOMCONFIGURATIONENTRY._serialized_options = b'8\001'
  _RECOGNIZEREQUEST._serialized_start=75
  _RECOGNIZEREQUEST._serialized_end=160
  _STREAMINGRECOGNIZEREQUEST._serialized_start=163
  _STREAMINGRECOGNIZEREQUEST._serialized_end=309
  _RECOGNITIONCONFIG._serialized_start=312
  _RECOGNITIONCONFIG._serialized_end=882
  _RECOGNITIONCONFIG_CUSTOMCONFIGURATIONENTRY._serialized_start=824
  _RECOGNITIONCONFIG_CUSTOMCONFIGURATIONENTRY._serialized_end=882
  _STREAMINGRECOGNITIONCONFIG._serialized_start=884
  _STREAMINGRECOGNITIONCONFIG._serialized_end=989
  _SPEECHCONTEXT._serialized_start=991
  _SPEECHCONTEXT._serialized_end=1038
  _RECOGNIZERESPONSE._serialized_start=1040
  _RECOGNIZERESPONSE._serialized_end=1118
  _SPEECHRECOGNITIONRESULT._serialized_start=1121
  _SPEECHRECOGNITIONRESULT._serialized_end=1261
  _SPEECHRECOGNITIONALTERNATIVE._serialized_start=1263
  _SPEECHRECOGNITIONALTERNATIVE._serialized_end=1375
  _WORDINFO._serialized_start=1377
  _WORDINFO._serialized_end=1459
  _STREAMINGRECOGNIZERESPONSE._serialized_start=1461
  _STREAMINGRECOGNIZERESPONSE._serialized_end=1551
  _STREAMINGRECOGNITIONRESULT._serialized_start=1554
  _STREAMINGRECOGNITIONRESULT._serialized_end=1734
  _RIVASPEECHRECOGNITION._serialized_start=1737
  _RIVASPEECHRECOGNITION._serialized_end=1963
# @@protoc_insertion_point(module_scope)
