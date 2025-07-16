import re
import os
import json
from dataclasses import dataclass
from typing import Dict, Any
from openai import OpenAI

@dataclass
class SOAPNoteResult:
    soap_markdown: str
    patient_name: str
    doctor_name: str
    file_path: str

class SOAPNoteAgent:
    """
    Agent to generate SOAP notes from transcript, clinical lists, and structured JSON
    using OpenRouter LLM (DeepSeek or similar).
    """

    def __init__(self, api_key: str, model: str = "deepseek/deepseek-r1-0528:free"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )
        self.model = model

    def generate_soap_note(self, transcript: str, clinical_lists: str, structured_json: Dict[str, Any]) -> SOAPNoteResult:
        """Creates a SOAP note from extracted medical data"""

        # Try extracting names from the transcript
        patient_name = self._extract_name(transcript, is_doctor=False) or "Unknown_Patient"
        doctor_name = self._extract_name(transcript, is_doctor=True) or "Unknown_Doctor"
        filename = f"SOAP_Note_{patient_name.replace(' ', '_')}.md"

        # Enhanced SOAP prompt with better medical formatting
        soap_prompt = f"""
You are a professional medical scribe creating a clinical SOAP note. Use the provided transcript and clinical data to generate a comprehensive, medically accurate SOAP note.

TRANSCRIPT (Patient-Doctor Conversation):
\"\"\"
{transcript}
\"\"\"

CLINICAL LISTS (Extracted Data):
\"\"\"
{clinical_lists}
\"\"\"

STRUCTURED JSON DATA:
{json.dumps(structured_json, indent=2)}

INSTRUCTIONS:
1. Generate a proper SOAP note using ONLY the information provided above
2. DO NOT fabricate any medical information not present in the data
3. Format as professional medical documentation
4. Use clear, clinical language appropriate for medical records
5. Include confidence indicators when information is unclear or missing

SOAP NOTE FORMAT:
- **Date:** [Today's date]
- **Patient:** [Extract from transcript or use "Name not provided"]
- **Provider:** [Extract from transcript or use "Provider not specified"]
- **Chief Complaint:** [Patient's primary concern in their own words from transcript]

**SUBJECTIVE:**
- Present the patient's symptoms, concerns, and history as described in the transcript
- Include relevant past medical history, current medications, and patient-reported symptoms
- Use direct quotes from patient when appropriate
- Note any psychosocial factors mentioned

**OBJECTIVE:**
- Document any physical examination findings mentioned
- Include vital signs, laboratory results, or diagnostic test results if provided
- Note any clinical observations made during the encounter
- If physical exam details are not provided, state "Physical examination not documented"

**ASSESSMENT:**
- Provide clinical impression/diagnosis based on the available information
- Include differential diagnoses if multiple conditions are being considered
- Explain the reasoning behind the assessment using the clinical data
- Rate confidence level if diagnosis is uncertain

**PLAN:**
- List recommended diagnostic tests or procedures
- Include medication changes or prescriptions
- Specify follow-up instructions and timeline
- Add patient education topics discussed
- Include any referrals or consultations recommended

IMPORTANT:
- Write in complete sentences, not bullet points
- Be medically precise and professional
- Indicate when information is not available rather than guessing
- Maintain patient confidentiality standards
- Use proper medical terminology

OUTPUT: Provide ONLY the formatted SOAP note in markdown. No additional commentary.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": soap_prompt}],
                temperature=0.1,
                max_tokens=2048
            )

            soap_markdown = response.choices[0].message.content.strip()

            # Save to file
            output_dir = "clinical_outputs"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(soap_markdown)

            print(f" SOAP note saved to: {file_path}")
            return SOAPNoteResult(
                soap_markdown=soap_markdown,
                patient_name=patient_name,
                doctor_name=doctor_name,
                file_path=file_path
            )

        except Exception as e:
            print(f" Failed to generate SOAP note: {e}")
            raise

    def _extract_name(self, transcript: str, is_doctor: bool = False) -> str:
        """Extract patient or doctor name from the transcript"""
        if is_doctor:
            # Look for various doctor title patterns
            patterns = [
                r"Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                r"Doctor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                r"I'm\s+Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                r"This is\s+Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
            ]
        else:
            # Look for patient name patterns
            patterns = [
                r"(?:Ms\.|Mrs\.|Mr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                r"My name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                r"I'm\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                r"Patient:\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
            ]
        
        for pattern in patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None


def complete_clinical_pipeline(audio_file_path: str, openrouter_api_key: str):
    """
    Complete pipeline: Audio → Transcript → Clinical Extraction → SOAP Note
    """
    from clinical_extraction_agent import ClinicalExtractionAgent  # Import your extraction agent
    
    # Step 1: Get transcript (you'll need to implement this or use existing code)
    transcript = get_transcript_from_audio(audio_file_path)  # Implement this function
    
    if not transcript:
        print(" Failed to get transcript")
        return None
    
    # Step 2: Extract clinical data
    extraction_agent = ClinicalExtractionAgent(openrouter_api_key)
    clinical_result = extraction_agent.extract_clinical_data(transcript)
    
    # Step 3: Generate SOAP note
    soap_agent = SOAPNoteAgent(api_key=openrouter_api_key)
    soap_result = soap_agent.generate_soap_note(
        transcript=transcript,  # This is the key fix - pass the actual transcript
        clinical_lists=clinical_result.clinical_lists,
        structured_json=clinical_result.structured_json
    )
    
    # Step 4: Save everything
    extraction_agent.save_outputs(clinical_result)
    
    return soap_result


def get_transcript_from_audio(audio_file_path: str) -> str:
    """
    Placeholder for audio transcription function
    Replace with your actual transcription code
    """
    # TODO: Implement your audio transcription logic here
    # This might use OpenAI Whisper, Google Speech-to-Text, etc.
    pass


if __name__ == "__main__":
    # Method 1: If you already have the transcript, clinical lists, and JSON
    
    # Load existing files (if you've already run the clinical extraction)
    try:
        with open("clinical_outputs/clinical_extracted_lists.txt", "r", encoding="utf-8") as f:
            clinical_lists = f.read()
        
        with open("clinical_outputs/clinical_structured_data.json", "r", encoding="utf-8") as f:
            structured_json = json.load(f)
        
        # YOU NEED TO LOAD THE ACTUAL TRANSCRIPT HERE
        # This is what was missing in your original code
        transcript = """
        [PUT YOUR ACTUAL TRANSCRIPT HERE]
        Doctor: Good morning, how are you feeling today?
        Patient: I've been having irregular periods and some pelvic discomfort...
        [etc.]
        """
        
        # Generate SOAP note
        api_key = "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"
        soap_agent = SOAPNoteAgent(api_key=api_key)
        
        result = soap_agent.generate_soap_note(
            transcript=transcript,
            clinical_lists=clinical_lists,
            structured_json=structured_json
        )
        
        print("\n FINAL SOAP NOTE:\n")
        print(result.soap_markdown)
        
    except FileNotFoundError:
        print(" Clinical extraction files not found. Run the clinical extraction first.")
    
    # Method 2: Complete pipeline from audio file
    # complete_clinical_pipeline("path/to/audio.mp3", "your-api-key")