import os
import json
import wave
import moviepy.editor as m_ed
from typing import Callable, Union
import whisperx
import cv2
import numpy as np
import torch
import librosa
import soundfile as sf
from TTS.api import TTS

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import sys
sys.path.append('/home/lindel/diploma')

from utils.audio_funcs import audioread, audiowrite, snr_mixer, audio_normalization, align_audio



class BaseAutoDubbingModel:
    def __init__(self):
        # Initialize models
        self.translate_tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-ru-en")
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-ru-en")
        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        
        
    def get_audio_from_video(self, input_video_file: str, output_audio_file: str) -> None:
        video = m_ed.VideoFileClip(input_video_file)
        audio = video.audio
        audio.write_audiofile(output_audio_file, ffmpeg_params=["-ac", "1"])
        print(f'Audio get successfully in {output_audio_file}')
        
    def get_text_from_voice(self, audio_file: str):
        return self.__get_text_from_voice_whisperx(audio_file)
    
    def translate(self, input_text_objs):
        output_text_objs = input_text_objs.copy()

        for idx, text_obj in enumerate(input_text_objs):
            input_text = text_obj.get('text')
            input_ids = self.translate_tokenizer.encode(input_text, return_tensors="pt")

            # Generate the translation
            output_ids = self.translate_model.generate(input_ids)

            # Decode the output tokens
            output_text = self.translate_tokenizer.decode(output_ids[0], skip_special_tokens=True)

            output_text_objs[idx]['text'] = output_text
        return output_text_objs
    
    def text_to_speech(self, output_text_objs, speaker_path, output_path, output_path_mod):
        self.tts_model.tts_to_file(text=' '.join([output_text['text'] for output_text in output_text_objs]),
                        file_path=output_path,
                        speaker_wav=speaker_path,
                        language="en")
        # Align length of translated audio
        real_audio, sr_r = audioread(speaker_path)
        translated_audio, sr = audioread(output_path)
        
        time_stretch_factor = len(translated_audio) / len(real_audio)
        modified_translated_audio = librosa.effects.time_stretch(translated_audio, rate=time_stretch_factor)
        
        audiowrite(modified_translated_audio, sr, output_path_mod)
        
    
        
    def __get_text_from_voice_whisperx(self, filename):
        device = "cuda"
        batch_size = 16 # reduce if low on GPU mem
        compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

        model = whisperx.load_model("large-v2", device, compute_type=compute_type,
                                    asr_options={'max_new_tokens':100, 'clip_timestamps':True, 'hallucination_silence_threshold':0.5})

        audio = whisperx.load_audio(filename)
        result = model.transcribe(audio, batch_size=batch_size)

        diarize_model = whisperx.DiarizationPipeline(use_auth_token='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', device=device)

        diarize_segments = diarize_model(filename)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        return result["segments"]


# UExample usage
base_input_path = '/input_path_dir'
base_output_path = '/output_path_dir'
input_video_path = f"{base_input_path}/input_video.mp4"
output_video_path = f"{base_output_path}/translated_video.mp4"
input_audio_path = f"{base_output_path}/input_audio.wav"
output_audio_path = f"{base_output_path}/translated_audio.wav"
output_mod_audio_path = f"{base_output_path}/translated_audio_mod.wav"

dubbing_model = BaseAutoDubbingModel()

dubbing_model.get_audio_from_video(input_video_path, input_audio_path)
input_text_objs = dubbing_model.get_text_from_voice(input_audio_path)
output_text_objs = dubbing_model.translate(input_text_objs)
dubbing_model.text_to_speech(output_text_objs, input_audio_path, output_audio_path, output_mod_audio_path)

