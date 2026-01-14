import re
import logging
import librosa
import numpy as np
from .prompts import PITCH_THRESHOLD 

logger = logging.getLogger(__name__)


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

def label_transcript_gender(transcript, audio_path: str) -> str:
    """
    Dựa trên transcript thô, tính toán F0 để gán nhãn [M] hoặc [F] cho từng người nói.
    """
    # Nếu đầu vào là thông báo lỗi từ bước trước, trả về luôn
    if isinstance(transcript, str): return transcript
    if not transcript.utterances: return "LỖI: Không tìm thấy nội dung hội thoại."

    # --- BƯỚC 1: XÁC ĐỊNH GIỚI TÍNH CỦA NGƯỜI NÓI ĐẦU TIÊN ---
    first_utterance = transcript.utterances[0]
    speaker_id = first_utterance.speaker
    
    # Tính tần số trung bình (F0) của đoạn hội thoại đầu tiên
    mean_f0 = get_mean_f0(audio_path, first_utterance.start, first_utterance.end)
    
    # Gán nhãn dựa trên ngưỡng Pitch (PITCH_THRESHOLD)
    # < 160Hz là Nam [M], ngược lại là Nữ [F]
    is_male = (0 < mean_f0 < PITCH_THRESHOLD)
    primary_label = "[M]" if is_male else "[F]"
    secondary_label = "[F]" if is_male else "[M]"
    gender_map = {speaker_id: primary_label}
    
    # --- BƯỚC 2: DUYỆT VÀ GHÉP NỘI DUNG HỘI THOẠI ---
    formatted_lines = []
    last_speaker = None

    for ut in transcript.utterances:
        # Nếu gặp speaker mới chưa có trong map, gán nhãn đối lập với speaker đầu tiên
        if ut.speaker not in gender_map:
            gender_map[ut.speaker] = secondary_label
            
        current_label = gender_map[ut.speaker]
        clean_text = ut.text.strip()

        if ut.speaker != last_speaker:
            # Nếu đổi người nói: Xuống dòng và thêm nhãn [M]/[F]
            formatted_lines.append(f"\n{current_label} {clean_text}")
        else:
            # Nếu cùng một người nói tiếp: Chỉ ghép thêm text vào dòng cũ
            formatted_lines.append(f" {clean_text}")
            
        last_speaker = ut.speaker

    return "".join(formatted_lines).strip()


def pre_validate_part3_context(problem_context: str) -> str:
    """Validate 4 đáp án (A)(B)(C)(D)."""
    if not problem_context: return "Bạn cung cấp không đủ dữ liệu."
    all_options = set(re.findall(r'[\(\s]([A-D])[\)\.]', problem_context.upper()))
    target_options = {'A', 'B', 'C','D'} 
    
    if len(all_options) == 4 and all_options == target_options:
        return "1"  
    else:
        return "Bạn cung cấp không đủ đáp án (A)(B)(C)(D) hoặc sai định dạng."

def extract_text(result) -> str:
    output = result.get("output", "")
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, list):
        return output[0].get("text", "").strip()
    return ""

def format_answer_only(text: str) -> str:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("Đáp án:"):
                return "\n".join(lines[i:]).strip()
        return text.strip()
