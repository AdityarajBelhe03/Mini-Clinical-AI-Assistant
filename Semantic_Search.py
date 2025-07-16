import os
from pathlib import Path

# Perform semantic search
search_results = embedding_agent.semantic_search(
    query="irregular periods and hormonal imbalance",
    top_k=3,
    medical_category="gynecology"  # Can be None for full search
)


output_dir = Path("search_results")
output_dir.mkdir(exist_ok=True)


for i, result in enumerate(search_results, 1):
    print(f"\nResult {i} (Score: {result['score']:.3f}):")
    print(f"Patient: {result['patient_name']}")
    print(f"Category: {result['medical_category']}")
    print(f"Section: {result['section']}")
    print(f"Diagnosis: {result['primary_diagnosis']}")
    print(f"Encounter Date: {result['encounter_date']}")
    print(f"Text:\n{result['text'][:300]}...\n")

    
    filename = output_dir / f"SearchResult_{i}_{result['medical_category']}_{result['section']}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Semantic Search Result {i}\n\n")
        f.write(f"**Score:** {result['score']:.3f}\n")
        f.write(f"**Patient Name:** {result['patient_name']}\n")
        f.write(f"**Medical Category:** {result['medical_category']}\n")
        f.write(f"**Section:** {result['section']}\n")
        f.write(f"**Primary Diagnosis:** {result['primary_diagnosis']}\n")
        f.write(f"**Encounter Date:** {result['encounter_date']}\n\n")
        f.write("---\n\n")
        f.write(result['text'])

print(f"\n All {len(search_results)} results saved to: {output_dir.absolute()}")