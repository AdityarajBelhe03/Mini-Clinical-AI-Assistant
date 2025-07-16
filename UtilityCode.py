import whisper
import requests
import json
import os
from typing import Dict, Any, Optional, List, TypedDict
from datetime import datetime
import numpy as np
from dataclasses import dataclass, field
import tempfile
import warnings
warnings.filterwarnings('ignore')

# STATE MANAGEMENT - Foundation for Multi-Agent System

class ClinicalState(TypedDict):
    """State object that flows through all agents"""
    # Core Data
    audio_file_path: Optional[str]
    raw_transcript: Optional[str]
    cleaned_transcript: Optional[str]
    clinical_entities: Optional[Dict[str, Any]]
    structured_json: Optional[Dict[str, Any]]
    soap_note: Optional[str]

    # RAG & Vector Components
    embeddings: Optional[List[float]]
    similar_cases: Optional[List[Dict[str, Any]]]
    enhanced_context: Optional[str]

    # Quality & Tracking
    confidence_scores: Optional[Dict[str, float]]
    processing_steps: List[str]
    errors: List[str]
    session_id: str

# AGENT 1: TRANSCRIPTION AGENT

class TranscriptionAgent:
    """
    Agent responsible for converting audio to text using Whisper AI
    - Uses local Whisper model (free and effective)
    - Handles various audio formats
    - Provides confidence scoring
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model

        Args:
            model_size: Options are "tiny", "base", "small", "medium", "large"
                       "base" is good balance of speed and accuracy
        """
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded successfully!")

    def transcribe_audio(self, audio_file_path: str, state: ClinicalState) -> ClinicalState:
        """
        Transcribe audio file to text

        Args:
            audio_file_path: Path to audio file
            state: Current clinical state

        Returns:
            Updated clinical state with transcript
        """
        try:
            print(f"Transcribing audio file: {audio_file_path}")

            # Update state
            state["audio_file_path"] = audio_file_path
            state["processing_steps"].append("transcription_started")

            # Transcribe using Whisper
            result = self.model.transcribe(audio_file_path)

            # Extract transcript and confidence
            transcript = result["text"].strip()

            # Whisper doesn't provide word-level confidence, so we estimate
            # based on number of segments and language detection confidence
            segments = result.get("segments", [])
            avg_confidence = sum(segment.get("no_speech_prob", 0.1) for segment in segments) / max(len(segments), 1)
            confidence = max(0.1, 1.0 - avg_confidence)  # Invert no_speech_prob

            # Update state
            state["raw_transcript"] = transcript
            state["confidence_scores"] = {"transcription": confidence}
            state["processing_steps"].append("transcription_completed")

            print(f"Transcription completed. Confidence: {confidence:.2f}")
            print(f"Transcript: {transcript[:200]}...")

            return state

        except Exception as e:
            error_msg = f"Transcription error: {str(e)}"
            state["errors"].append(error_msg)
            print(f"Error: {error_msg}")
            return state

    def create_sample_audio_text(self, text: str, output_path: str = "sample_audio.txt") -> str:
        """
        Create a sample text file simulating audio transcription
        (Useful for testing when no actual audio file is available)
        """
        with open(output_path, 'w') as f:
            f.write(text)
        return output_path

# UTILITY FUNCTIONS

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

def save_state_to_file(state: ClinicalState, filename: str = None):
    """Save current state to JSON file"""
    if filename is None:
        filename = f"clinical_state_{state['session_id']}.json"

    # Convert state to serializable format
    serializable_state = {k: v for k, v in state.items() if v is not None}

    with open(filename, 'w') as f:
        json.dump(serializable_state, f, indent=2)

    print(f"State saved to {filename}")