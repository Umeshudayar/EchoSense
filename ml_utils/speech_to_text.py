import os
import whisper
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import tempfile
import logging
from config import WHISPER_MODEL, VOSK_MODEL_PATH

logger = logging.getLogger(__name__)

class SpeechToTextConverter:
    def __init__(self, model_type="whisper"):
        """Initialize the speech-to-text converter with either Whisper or Vosk."""
        self.model_type = model_type
        
        if model_type == "whisper":
            logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
            self.model = whisper.load_model(WHISPER_MODEL)
        elif model_type == "vosk":
            if not os.path.exists(VOSK_MODEL_PATH):
                raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}")
            logger.info(f"Loading Vosk model from: {VOSK_MODEL_PATH}")
            self.model = Model(VOSK_MODEL_PATH)
        else:
            raise ValueError("Model type must be either 'whisper' or 'vosk'")
    
    def convert_to_wav(self, audio_path):
        """Convert any audio format to WAV for Vosk processing."""
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set frame rate to 16kHz
        audio.export(temp_wav, format="wav")
        return temp_wav
    
    def transcribe_whisper(self, audio_path):
        """Transcribe audio using OpenAI's Whisper model."""
        try:
            result = self.model.transcribe(audio_path)
            return {
                "text": result["text"],
                "segments": [
                    {
                        "text": segment["text"],
                        "start": segment["start"],
                        "end": segment["end"]
                    } for segment in result["segments"]
                ]
            }
        except Exception as e:
            logger.error(f"Error transcribing with Whisper: {e}")
            raise
    
    def transcribe_vosk(self, audio_path):
        """Transcribe audio using Vosk model."""
        wav_path = self.convert_to_wav(audio_path)
        
        try:
            wf = wave.open(wav_path, "rb")
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                logger.error("Audio file must be WAV format mono PCM.")
                return {"text": "Error: Incompatible audio format"}
            
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)
            
            full_text = ""
            segments = []
            
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if "text" in result:
                        full_text += " " + result["text"]
                        segments.append({
                            "text": result["text"],
                            "start": result.get("start", 0),
                            "end": result.get("end", 0)
                        })
            
            final_result = json.loads(rec.FinalResult())
            if "text" in final_result:
                full_text += " " + final_result["text"]
                segments.append({
                    "text": final_result["text"],
                    "start": final_result.get("start", 0),
                    "end": final_result.get("end", 0)
                })
            
            return {
                "text": full_text.strip(),
                "segments": segments
            }
            
        except Exception as e:
            logger.error(f"Error transcribing with Vosk: {e}")
            raise
        finally:
            # Clean up temporary WAV file
            if os.path.exists(wav_path):
                os.remove(wav_path)
    
    def transcribe(self, audio_path):
        """Transcribe audio using the selected model."""
        if self.model_type == "whisper":
            return self.transcribe_whisper(audio_path)
        else:
            return self.transcribe_vosk(audio_path)