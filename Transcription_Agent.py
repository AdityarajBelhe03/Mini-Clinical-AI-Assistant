import whisper
import json
from typing import Dict, Any, Optional, List, TypedDict
from datetime import datetime
import os


class ClinicalState(TypedDict):
    """State object that flows through all agents"""
    audio_file_path: Optional[str]
    raw_transcript: Optional[str]
    cleaned_transcript: Optional[str]
    clinical_entities: Optional[Dict[str, Any]]
    structured_json: Optional[Dict[str, Any]]
    soap_note: Optional[str]
    embeddings: Optional[List[float]]
    similar_cases: Optional[List[Dict[str, Any]]]
    enhanced_context: Optional[str]
    confidence_scores: Optional[Dict[str, float]]
    processing_steps: List[str]
    errors: List[str]
    session_id: str

def initialize_clinical_state(session_id: str = None) -> ClinicalState:
    """Initialize a new clinical state"""
    if session_id is None:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return ClinicalState(
        audio_file_path=None,
        raw_transcript=None,
        cleaned_transcript=None,
        clinical_entities=None,
        structured_json=None,
        soap_note=None,
        embeddings=None,
        similar_cases=None,
        enhanced_context=None,
        confidence_scores=None,
        processing_steps=[],
        errors=[],
        session_id=session_id
    )

# TRANSCRIPTION AGENT

class TranscriptionAgent:
    """Agent responsible for converting audio to text using Whisper AI"""

    def __init__(self, model_size: str = "base"):
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded successfully!")

    def transcribe_audio(self, audio_file_path: str, state: ClinicalState) -> ClinicalState:
        """Transcribe audio file to text"""
        try:
            print(f"Transcribing audio file: {audio_file_path}")

            # Check if file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

            # Update state
            state["audio_file_path"] = audio_file_path
            state["processing_steps"].append("transcription_started")

            # Transcribe using Whisper
            result = self.model.transcribe(audio_file_path)

            # Extract transcript
            transcript = result["text"].strip()

            # Calculate confidence (Whisper doesn't provide direct confidence)
            segments = result.get("segments", [])
            if segments:
                avg_confidence = sum(segment.get("no_speech_prob", 0.1) for segment in segments) / len(segments)
                confidence = max(0.1, 1.0 - avg_confidence)
            else:
                confidence = 0.8  # Default confidence

            # Update state
            state["raw_transcript"] = transcript
            state["confidence_scores"] = {"transcription": confidence}
            state["processing_steps"].append("transcription_completed")

            print(f"✅ Transcription completed. Confidence: {confidence:.2f}")
            print(f"Transcript: {transcript}")

            return state

        except Exception as e:
            error_msg = f"Transcription error: {str(e)}"
            state["errors"].append(error_msg)
            print(f"❌ Error: {error_msg}")
            return state


def test_audio_to_text(audio_file_path: str):
    """
    Simple function to test audio-to-text conversion

    Args:
        audio_file_path: Path to your .mp3, .wav, or other audio file

    Returns:
        transcript: The extracted text from audio
    """

    print(f"Testing audio file: {audio_file_path}")
    print("Loading Whisper model...")

    # Initialize the transcription agent
    transcription_agent = TranscriptionAgent(model_size="base")

    # Initialize state
    state = initialize_clinical_state()

    # Process the audio file
    print("Processing audio...")
    state = transcription_agent.transcribe_audio(audio_file_path, state)

    # Show results
    if state["errors"]:
        print("❌ Errors occurred:")
        for error in state["errors"]:
            print(f"  - {error}")
        return None

    print("✅ Transcription successful!")
    print(f"Transcript: {state['raw_transcript']}")
    print(f"Confidence: {state['confidence_scores']['transcription']:.2f}")

    return state["raw_transcript"]

# MAIN TESTING FUNCTION

if __name__ == "__main__":
    print("=== AUDIO-TO-TEXT TESTING ===")
    print("Usage:")
    print("1. test_audio_to_text('path/to/your/audio.mp3')")

    transcript = test_audio_to_text("/content/ElevenLabs_2025-07-15T19_33_20_Liam_pre_sp100_s50_sb100_v3.mp3")
    # print(f"Final transcript: {transcript}")