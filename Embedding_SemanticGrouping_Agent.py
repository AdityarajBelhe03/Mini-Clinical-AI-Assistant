import os
import json
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone


@dataclass
class SOAPEmbeddingResult:
    """Result of SOAP embedding process"""
    document_id: str
    total_chunks: int
    medical_category: str
    patient_name: str
    doctor_name: str
    upserted_vectors: int
    processing_time: float


class SOAPEmbeddingAgent:
    """
    Agent to convert SOAP notes into embeddings and upsert them to Pinecone
    with semantic grouping and medical categorization for efficient RAG.
    """
    
    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str = "newclinicai",
        embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2",  
        namespace: str = "default"
    ):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Get embedding dimension for validation
        test_embedding = self.embedding_model.embed_query("test")
        self.embedding_dimension = len(test_embedding)
        print(f" Embedding model dimension: {self.embedding_dimension}")
        print(f" Model: {embedding_model}")
        print(f" Pinecone index: {index_name}")
        print(f" Namespace: {namespace}")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        
        # Text splitter for creating semantic chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", ".", " ", ""]
        )
        
        # Medical category mapping for semantic grouping
        self.medical_categories = {
            "gynecology": ["pcos", "menstrual", "irregular periods", "ovarian", "gynecological", 
                          "pelvic", "hormonal", "fertility", "reproductive", "estrogen", "testosterone"],
            "cardiology": ["heart", "cardiac", "blood pressure", "chest pain", "arrhythmia", 
                          "cardiovascular", "ecg", "ekg", "coronary", "hypertension"],
            "endocrinology": ["diabetes", "thyroid", "insulin", "glucose", "hormone", "metabolic", 
                             "endocrine", "cortisol", "adrenal", "pituitary"],
            "neurology": ["headache", "migraine", "seizure", "stroke", "neurological", "brain", 
                         "nervous system", "neuropathy", "dementia", "parkinson"],
            "dermatology": ["skin", "rash", "acne", "dermatitis", "eczema", "psoriasis", 
                           "dermatological", "lesion", "mole", "hair loss"],
            "gastroenterology": ["stomach", "digestive", "gastric", "intestinal", "bowel", 
                               "liver", "pancreas", "nausea", "vomiting", "diarrhea"],
            "respiratory": ["lung", "respiratory", "asthma", "copd", "pneumonia", "bronchitis", 
                           "dyspnea", "cough", "breathing", "pulmonary"],
            "orthopedics": ["bone", "joint", "fracture", "arthritis", "musculoskeletal", 
                           "orthopedic", "spine", "muscle", "tendon", "ligament"],
            "psychiatry": ["depression", "anxiety", "psychiatric", "mental health", "mood", 
                          "bipolar", "schizophrenia", "therapy", "psychological", "stress"],
            "general_medicine": ["general", "primary care", "routine", "check-up", "wellness", 
                               "preventive", "screening", "vaccination", "physical exam"]
        }

    def generate_id(self, text: str, patient_name: str = "", timestamp: str = "") -> str:
        """Generate unique hash-based ID for documents"""
        combined = f"{text}_{patient_name}_{timestamp}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def categorize_medical_condition(self, soap_content: str) -> str:
        """
        Categorize SOAP note into medical specialty based on content analysis
        This enables semantic grouping for better RAG performance
        """
        soap_lower = soap_content.lower()
        category_scores = {}
        
        # Score each category based on keyword matches
        for category, keywords in self.medical_categories.items():
            score = sum(1 for keyword in keywords if keyword in soap_lower)
            # Weight certain sections more heavily
            if "assessment:" in soap_lower:
                assessment_section = soap_lower.split("assessment:")[1].split("plan:")[0]
                score += sum(2 for keyword in keywords if keyword in assessment_section)
            
            category_scores[category] = score
        
        # Return category with highest score, default to general_medicine
        best_category = max(category_scores, key=category_scores.get)
        return best_category if category_scores[best_category] > 0 else "general_medicine"

    def extract_metadata_from_soap(self, soap_content: str) -> Dict[str, Any]:
        """Extract structured metadata from SOAP note content"""
        metadata = {}
        
        # Extract patient name
        patient_match = re.search(r"Patient:\s*([^\n]+)", soap_content)
        metadata["patient_name"] = patient_match.group(1).strip() if patient_match else "Unknown"
        
        # Extract provider name  
        provider_match = re.search(r"Provider:\s*([^\n]+)", soap_content)
        metadata["provider_name"] = provider_match.group(1).strip() if provider_match else "Unknown"
        
        # Extract date
        date_match = re.search(r"Date:\s*([^\n]+)", soap_content)
        metadata["encounter_date"] = date_match.group(1).strip() if date_match else "Unknown"
        
        # Extract chief complaint
        cc_match = re.search(r"Chief Complaint:\s*([^\n]+)", soap_content)
        metadata["chief_complaint"] = cc_match.group(1).strip() if cc_match else "Unknown"
        
        # Extract assessment/diagnosis
        assessment_match = re.search(r"ASSESSMENT:\s*(.*?)(?=PLAN:|$)", soap_content, re.DOTALL)
        if assessment_match:
            assessment_text = assessment_match.group(1).strip()
            # Extract primary diagnosis
            primary_dx_match = re.search(r"Primary Diagnosis:\s*([^\n]+)", assessment_text)
            metadata["primary_diagnosis"] = primary_dx_match.group(1).strip() if primary_dx_match else "Unknown"
        
        return metadata

    def create_document_chunks(self, soap_content: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Create semantic chunks from SOAP note with section-aware splitting
        """
        # Split SOAP note into sections
        sections = {
            "subjective": re.search(r"SUBJECTIVE:\s*(.*?)(?=OBJECTIVE:|$)", soap_content, re.DOTALL),
            "objective": re.search(r"OBJECTIVE:\s*(.*?)(?=ASSESSMENT:|$)", soap_content, re.DOTALL),
            "assessment": re.search(r"ASSESSMENT:\s*(.*?)(?=PLAN:|$)", soap_content, re.DOTALL),
            "plan": re.search(r"PLAN:\s*(.*?)$", soap_content, re.DOTALL)
        }
        
        documents = []
        
        # Process each section separately for better semantic coherence
        for section_name, section_match in sections.items():
            if section_match:
                section_content = section_match.group(1).strip()
                
                # Create chunks for this section
                section_chunks = self.text_splitter.split_text(section_content)
                
                for i, chunk in enumerate(section_chunks):
                    # Enhanced metadata for each chunk
                    chunk_metadata = {
                        **metadata,
                        "section": section_name,
                        "chunk_index": i,
                        "total_chunks_in_section": len(section_chunks),
                        "section_content": section_content[:200] + "..." if len(section_content) > 200 else section_content
                    }
                    
                    # Create document with section header for context
                    document_text = f"SOAP {section_name.upper()} Section:\n{chunk}"
                    
                    documents.append(Document(
                        page_content=document_text,
                        metadata=chunk_metadata
                    ))
        
        # Also create a summary chunk with key information
        summary_text = f"""
        Medical Summary:
        Patient: {metadata.get('patient_name', 'Unknown')}
        Provider: {metadata.get('provider_name', 'Unknown')}
        Chief Complaint: {metadata.get('chief_complaint', 'Unknown')}
        Primary Diagnosis: {metadata.get('primary_diagnosis', 'Unknown')}
        Medical Category: {metadata.get('medical_category', 'Unknown')}
        """
        
        summary_metadata = {
            **metadata,
            "section": "summary",
            "chunk_index": 0,
            "is_summary": True
        }
        
        documents.append(Document(
            page_content=summary_text.strip(),
            metadata=summary_metadata
        ))
        
        return documents

    def validate_index_dimension(self) -> bool:
        """
        Validate that embedding dimension matches Pinecone index dimension
        """
        try:
            index_stats = self.index.describe_index_stats()
            index_dimension = index_stats['dimension']
            
            print(f"  DIMENSION CHECK:")
            print(f"   Pinecone index dimension: {index_dimension}")
            print(f"   Embedding model dimension: {self.embedding_dimension}")
            
            if index_dimension != self.embedding_dimension:
                print(f"  DIMENSION MISMATCH!")
                print(f"   Your Pinecone index expects {index_dimension}D vectors")
                print(f"   But your embedding model produces {self.embedding_dimension}D vectors")
                print(f"\n SOLUTIONS:")
                print(f"   1. Use a different embedding model:")
                
                # Suggest compatible models based on index dimension
                model_suggestions = {
                    384: ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/paraphrase-MiniLM-L3-v2"],
                    768: ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-roberta-large-v1"],
                    1024: ["sentence-transformers/all-MiniLM-L12-v2", "BAAI/bge-large-en-v1.5"],
                    1536: ["text-embedding-ada-002 (OpenAI)", "text-embedding-3-small (OpenAI)"]
                }
                
                if index_dimension in model_suggestions:
                    for model in model_suggestions[index_dimension]:
                        print(f"      - {model}")
                else:
                    print(f"      - Create a new index with {self.embedding_dimension} dimensions")
                
                print(f"   2. OR create a new Pinecone index with {self.embedding_dimension} dimensions")
                return False
            else:
                print(f" Dimensions match! Ready to proceed.")
                return True
                
        except Exception as e:
            print(f" Error validating dimensions: {e}")
            return False

    def process_soap_file(self, file_path: str) -> SOAPEmbeddingResult:
        """
        Process a single SOAP note file and upsert embeddings to Pinecone
        """
        start_time = datetime.now()
        
        # Read SOAP note content
        with open(file_path, 'r', encoding='utf-8') as f:
            soap_content = f.read()
        
        # Extract metadata
        metadata = self.extract_metadata_from_soap(soap_content)
        
        # Categorize medical condition
        medical_category = self.categorize_medical_condition(soap_content)
        metadata["medical_category"] = medical_category
        metadata["file_path"] = file_path
        metadata["processing_timestamp"] = datetime.now().isoformat()
        
        # Create document chunks
        documents = self.create_document_chunks(soap_content, metadata)
        
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector_id = self.generate_id(
                doc.page_content,
                metadata.get("patient_name", ""),
                f"{i}_{datetime.now().isoformat()}"
            )
            
            # Enhanced metadata for semantic search - CRUCIAL: include "text" field for RAG
            vector_metadata = {
                **doc.metadata,
                "text": doc.page_content, 
                "embedding_model": self.embedding_model.model_name,  
                "vector_id": vector_id,
                "embedding_dimension": len(embedding)
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": vector_metadata
            })
        
        # Upsert to Pinecone with medical category namespace for grouping
        category_namespace = f"{self.namespace}_{medical_category}"
        self.index.upsert(vectors=vectors_to_upsert, namespace=category_namespace)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"    Processed SOAP note: {Path(file_path).name}")
        print(f"    Medical Category: {medical_category}")
        print(f"    Patient: {metadata.get('patient_name', 'Unknown')}")
        print(f"    Diagnosis: {metadata.get('primary_diagnosis', 'Unknown')}")
        print(f"    Chunks created: {len(documents)}")
        print(f"    Vectors upserted: {len(vectors_to_upsert)}")
        print(f"    Namespace: {category_namespace}")
        print(f"    Processing time: {processing_time:.2f}s")
        
        return SOAPEmbeddingResult(
            document_id=vectors_to_upsert[0]["id"],
            total_chunks=len(documents),
            medical_category=medical_category,
            patient_name=metadata.get("patient_name", "Unknown"),
            doctor_name=metadata.get("provider_name", "Unknown"),
            upserted_vectors=len(vectors_to_upsert),
            processing_time=processing_time
        )

    def process_all_soap_files(self, directory_path: str = "clinical_outputs") -> List[SOAPEmbeddingResult]:
        """
        Process all SOAP note files in the specified directory
        """
        # Validate dimensions before processing
        if not self.validate_index_dimension():
            print(" Cannot proceed due to dimension mismatch. Please fix the issue above.")
            return []
            
        results = []
        soap_files = list(Path(directory_path).glob("SOAP_Note_*.md"))
        
        if not soap_files:
            print(f" No SOAP note files found in {directory_path}")
            return results
        
        print(f" Processing {len(soap_files)} SOAP note files...")
        
        for file_path in soap_files:
            try:
                result = self.process_soap_file(str(file_path))
                results.append(result)
                print()  # Add spacing between files
            except Exception as e:
                print(f" Error processing {file_path}: {e}")
                continue
        
        # Summary statistics
        if results:  #  Fix division by zero error
            total_vectors = sum(r.upserted_vectors for r in results)
            total_time = sum(r.processing_time for r in results)
            categories = list(set(r.medical_category for r in results))
            
            print(f"	 PROCESSING SUMMARY:")
            print(f"   Files processed: {len(results)}")
            print(f"   Total vectors: {total_vectors}")
            print(f"   Medical categories: {', '.join(categories)}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average time per file: {total_time/len(results):.2f}s")
        else:
            print("PROCESSING SUMMARY: No files were successfully processed.")
        
        return results

    def semantic_search(self, query: str, top_k: int = 5, medical_category: str = None) -> List[Dict]:
        """
        Perform semantic search on stored SOAP notes
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Determine namespace for search
        if medical_category:
            namespace = f"{self.namespace}_{medical_category}"
        else:
            namespace = self.namespace
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'id': match['id'],
                'score': match['score'],
                'patient_name': match['metadata'].get('patient_name', 'Unknown'),
                'medical_category': match['metadata'].get('medical_category', 'Unknown'),
                'section': match['metadata'].get('section', 'Unknown'),
                'primary_diagnosis': match['metadata'].get('primary_diagnosis', 'Unknown'),
                'text': match['metadata'].get('text', ''),
                'encounter_date': match['metadata'].get('encounter_date', 'Unknown')
            })
        
        return formatted_results

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        stats = self.index.describe_index_stats()
        return {
            'total_vectors': stats['total_vector_count'],
            'namespaces': stats.get('namespaces', {}),
            'dimension': stats['dimension']
        }


# Usage Example and Testing
if __name__ == "__main__":
    # Initialize the embedding agent
    PINECONE_API_KEY = "pcsk_2p56Ab_D4bPrEbd2WZ1CorhConZvSAXHfX1WLu4nsbm149XhFZuFTT9oK5y8KbSYCCghj3"  # Replace with your actual API key
    
    try:
        # Create embedding agent with 1024D model to match your index
        embedding_agent = SOAPEmbeddingAgent(
            pinecone_api_key=PINECONE_API_KEY,
            index_name="newclinicai",
            embedding_model="sentence-transformers/all-MiniLM-L12-v2",  
            namespace="soap_docs"
        )
        
        # Process all SOAP files automatically
        results = embedding_agent.process_all_soap_files("clinical_outputs")
        
        # Only proceed with search if processing was successful
        if results:
            # Display index statistics
            print("\n INDEX STATISTICS:")
            stats = embedding_agent.get_index_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            # Example semantic search
            print("\n SEMANTIC SEARCH EXAMPLE:")
            search_results = embedding_agent.semantic_search(
                query="irregular periods and hormonal imbalance",
                top_k=3,
                medical_category="gynecology"
            )
            
            for i, result in enumerate(search_results, 1):
                print(f"\n   Result {i} (Score: {result['score']:.3f}):")
                print(f"   Patient: {result['patient_name']}")
                print(f"   Category: {result['medical_category']}")
                print(f"   Section: {result['section']}")
                print(f"   Text: {result['text'][:200]}...")
        else:
            print(" No files were processed successfully.")
            
    except Exception as e:
        print(f" Error: {e}")
        print("Please ensure you have the required packages installed:")
        print("pip install langchain.")