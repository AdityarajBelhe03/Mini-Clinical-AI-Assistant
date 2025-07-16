import json
import re
from typing import Dict, Any, Tuple
from openai import OpenAI
from dataclasses import dataclass
import os


@dataclass
class ClinicalExtractionResult:
    """Result container for clinical extraction"""
    clinical_lists: str
    structured_json: Dict[str, Any]
    confidence_scores: Dict[str, float]
    raw_extraction: str


class ClinicalExtractionAgent:
    """Combined Clinical Extraction & JSON Mapping Agent"""

    def __init__(self, openrouter_api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key
        )
        self.model = "deepseek/deepseek-r1-0528:free"

    def extract_clinical_data(self, transcript: str) -> ClinicalExtractionResult:
        """Extract clinical information and map to JSON with confidence scores"""
        print(" Sending transcript to LLM for clinical list extraction...")
        clinical_lists = self._extract_clinical_lists(transcript)
        print(" Got clinical lists!\n")

        print(" Sending clinical lists to LLM for JSON mapping...")
        structured_json, confidence_scores = self._map_to_json(clinical_lists, transcript)
        print(" Got structured JSON!\n")

        return ClinicalExtractionResult(
            clinical_lists=clinical_lists,
            structured_json=structured_json,
            confidence_scores=confidence_scores,
            raw_extraction=transcript
        )

    def _extract_clinical_lists(self, transcript: str) -> str:
        """Extract clinical terms into categorized lists"""
        extraction_prompt = f"""
You are a medical AI assistant specialized in extracting clinical information from medical transcripts.

TASK: Extract ONLY the clinical terms mentioned in the transcript and organize them into the following categories. Do NOT generate or infer information not explicitly stated.

TRANSCRIPT: "{transcript}"

INSTRUCTIONS:
1. Extract ONLY clinical terms, words, or phrases that are explicitly mentioned
2. Do NOT include full sentences or explanations
3. Do NOT generate or infer information not present in the transcript
4. If a category has no information, write "None mentioned"
5. Keep extractions concise - only the essential clinical terms

EXTRACT INTO THESE CATEGORIES:

**DEMOGRAPHICS:**
- Age: [extract only if mentioned]
- Sex: [extract only if mentioned]

**CHIEF COMPLAINT:**
- [list only the complaints/symptoms mentioned]

**PAST MEDICAL HISTORY:**
- [list only if mentioned]

**RECOMMENDED TESTS:**
- [list only tests explicitly recommended]

**CLINICAL FINDINGS/LAB VALUES:**
- [list only findings/values mentioned]

**DIAGNOSIS:**
- [list only diagnoses mentioned]

**MEDICATIONS:**
- [list only medications with doses/frequencies mentioned]

**CONSULTING PHYSICIAN:**
- [extract only if mentioned]

OUTPUT FORMAT: Present as clear lists under each category header.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f" Error in clinical extraction: {e}")
            return "Error in extraction"

    def _map_to_json(self, clinical_lists: str, transcript: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Map clinical lists to structured JSON with confidence scoring"""

        json_schema_description = """
{
  "clinical_data": {
    "age": integer | null,
    "sex": string | null,
    "chief_complaint": string | null,
    "diagnosis": string | null,
    "recommended_tests": [string],
    "medications": [
      {
        "name": string,
        "dose": string,
        "frequency": string
      }
    ]
  },
  "confidence_scores": {
    "age": float,
    "sex": float,
    "chief_complaint": float,
    "diagnosis": float,
    "recommended_tests": float,
    "medications": float
  }
}
"""

        json_mapping_prompt = f'''
You are a medical AI assistant. You are given:

1. A medical transcript
2. A list of extracted clinical terms from that transcript

Your job is to fill in the following JSON schema based ONLY on the given data — no assumptions, no invented content.

Transcript:
"{transcript}"

Extracted Clinical Lists:
"{clinical_lists}"

Follow this JSON schema exactly — do not alter keys or structure:

{json_schema_description}

Respond ONLY with valid JSON.
'''

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": json_mapping_prompt}],
                temperature=0.1,
                max_tokens=2048
            )

            raw_response = response.choices[0].message.content.strip()
            print(" Raw LLM JSON Output:\n", raw_response)

            raw_response = re.sub(r"```(?:json)?", "", raw_response).strip()
            match = re.search(r"\{[\s\S]*\}", raw_response)
            if not match:
                raise ValueError("No JSON object found in response.")

            clean_json_text = match.group(0)

            if not clean_json_text.endswith("}"):
                print(" Truncated JSON, attempting to close.")
                clean_json_text += "}}"

            print(" Extracted JSON to parse:\n", clean_json_text)
            parsed_response = json.loads(clean_json_text)

            clinical_data = parsed_response.get("clinical_data", {})
            confidence_scores = parsed_response.get("confidence_scores", {})

            return clinical_data, confidence_scores

        except Exception as e:
            print(f" Error in JSON mapping: {e}")
            print(" Raw text received (unparsed):\n", raw_response if 'raw_response' in locals() else "N/A")

            empty_json = {
                "age": None,
                "sex": None,
                "chief_complaint": None,
                "diagnosis": None,
                "recommended_tests": [],
                "medications": []
            }
            empty_confidence = {
                "age": 0.0,
                "sex": 0.0,
                "chief_complaint": 0.0,
                "diagnosis": 0.0,
                "recommended_tests": 0.0,
                "medications": 0.0
            }
            return empty_json, empty_confidence

    def save_outputs(self, result: ClinicalExtractionResult, output_dir: str = "clinical_outputs"):
        """Save clinical lists and JSON to files"""
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "clinical_extracted_lists.txt"), 'w', encoding='utf-8') as f:
            f.write("CLINICAL EXTRACTION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(result.clinical_lists)
            f.write("\n\n" + "=" * 50 + "\n")
            f.write("CONFIDENCE SCORES:\n")
            for field, score in result.confidence_scores.items():
                f.write(f"- {field}: {score:.2f}\n")

        with open(os.path.join(output_dir, "clinical_structured_data.json"), 'w', encoding='utf-8') as f:
            json.dump(result.structured_json, f, indent=2)

        with open(os.path.join(output_dir, "confidence_scores.json"), 'w', encoding='utf-8') as f:
            json.dump(result.confidence_scores, f, indent=2)

        print(" Files saved successfully!")


# === EXECUTION ===
def main():
    # === API Key and Audio Path ===
    OPENROUTER_API_KEY = "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"
    AUDIO_FILE = "/content/ElevenLabs_2025-07-15T19_33_20_Liam_pre_sp100_s50_sb100_v3.mp3"

    # === STEP 1: Transcribe audio using Agent 1 ===
    
    transcript = test_audio_to_text(AUDIO_FILE)

    if not transcript:
        print("❌ Transcript generation failed. Aborting.")
        return

    # === STEP 2: Extract structured data using Agent 2 ===
    agent = ClinicalExtractionAgent(OPENROUTER_API_KEY)
    print(" Processing clinical transcript...")
    result = agent.extract_clinical_data(transcript)

    # === STEP 3: Print Results ===
    print("\n CLINICAL LISTS:\n", result.clinical_lists)
    print("\n STRUCTURED JSON:\n", json.dumps(result.structured_json, indent=2))
    print("\n CONFIDENCE SCORES:")
    for field, score in result.confidence_scores.items():
        print(f"- {field}: {score:.2f}")

    # === STEP 4: Save Outputs ===
    print("\n Saving outputs...")
    agent.save_outputs(result)


if __name__ == "__main__":
    main()