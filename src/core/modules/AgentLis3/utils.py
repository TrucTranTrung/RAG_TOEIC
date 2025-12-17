import os
import re
import logging
import assemblyai as aai
import librosa
import numpy as np
from dotenv import load_dotenv
from .prompts import PITCH_THRESHOLD 

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="config/.env")
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY

def get_mean_f0(audio_file, start_ms, end_ms):
    """Tính Tần số Cơ bản (F0) trung bình của đoạn âm thanh."""
    try:
        y, sr = librosa.load(audio_file, sr=None)
        start_sample = int(start_ms * sr / 1000)
        end_sample = int(end_ms * sr / 1000)
        segment = y[start_sample:end_sample]
        f0, voiced_flag, _ = librosa.pyin(
            segment, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        valid_f0 = f0[voiced_flag]
        return np.mean(valid_f0) if valid_f0.size > 0 else 0.0
    except Exception as e:
        logger.error(f"Lỗi Librosa: {e}")
        return 0.0

def process_and_label_gender(audio_path):
    """Xử lý Diarization và gán nhãn giới tính [M]/[F]."""
    if not ASSEMBLYAI_API_KEY:
         return "LỖI CẤU HÌNH: AssemblyAI API Key không được tìm thấy."
         
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True)
    
    try:
        transcript = transcriber.transcribe(audio_path, config=config)
    except Exception as e:
        return f"LỖI KẾT NỐI/UPLOAD AAI: {e}"

    if transcript.status == aai.TranscriptStatus.error or not transcript.utterances:
        return f"LỖẼ XỬ LÝ AAI: {transcript.error if transcript.status == aai.TranscriptStatus.error else 'Transcript rỗng'}"

    first_u = transcript.utterances[0]
    speaker_a_id = first_u.speaker
    mean_f0_a = get_mean_f0(audio_path, first_u.start, first_u.end)
    gender_a_label = "[M]" if mean_f0_a < PITCH_THRESHOLD and mean_f0_a > 0 else "[F]" 

    if gender_a_label == "[M]":
        gender_map = {speaker_a_id: "[M]", "B": "[F]"} 
    else:
        gender_map = {speaker_a_id: "[F]", "B": "[M]"}
        
    logger.info(f"DEBUG: Speaker {speaker_a_id} F0: {mean_f0_a:.2f} Hz -> Nhãn: {gender_map.get(speaker_a_id)}")

    final_labeled_transcript = ""
    previous_speaker = None
    for utterance in transcript.utterances:
        current_speaker_id = utterance.speaker
        gender_label = gender_map.get(current_speaker_id, "[?]")
        
        if current_speaker_id != previous_speaker:
            final_labeled_transcript += f"\n{gender_label} {utterance.text.strip()}"
        else:
            final_labeled_transcript += f" {utterance.text.strip()}"
        previous_speaker = current_speaker_id

    return final_labeled_transcript.strip()


def pre_validate_part3_context(problem_context: str) -> str:
    """Validate 4 đáp án (A)(B)(C)(D)."""
    if not problem_context: return "Bạn cung cấp không đủ dữ liệu."
    all_options = set(re.findall(r'[\(\s]([A-D])[\)\.]', problem_context.upper()))
    target_options = {'A', 'B', 'C','D'} 
    
    if len(all_options) == 4 and all_options == target_options:
        return "1"  
    else:
        return "Bạn cung cấp không đủ đáp án (A)(B)(C)(D) hoặc sai định dạng."