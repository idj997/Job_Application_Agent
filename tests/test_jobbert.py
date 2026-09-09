import torch
from transformers import pipeline
import re
from app.classifiers.importance_rule import classify_skills


MODEL_NAME = "jjzha/jobbert_knowledge_extraction"

device = 0 if torch.cuda.is_available() else -1

print("Using device:", "GPU" if device == 0 else "CPU")
print("Loading model:", MODEL_NAME)


extractor = pipeline(
    task="token-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    aggregation_strategy="simple",
    device=device,
)


job_description = """
We are looking for an experienced Senior Data Engineer.

You will build scalable data pipelines using Azure Databricks,
PySpark and Delta Lake.

Strong SQL experience is essential.

Experience with CI/CD and Azure DevOps is required.

Experience with Apache Airflow would be desirable.
"""


# results = extractor(job_description)


# print("\nDetected entities:\n")

# for entity in results:
#     print(
#         f"Text:       {entity['word']}"
#     )
#     print(
#         f"Label:      {entity['entity_group']}"
#     )
#     print(
#         f"Confidence: {entity['score']:.4f}"
#     )
#     print(
#         f"Span:       {entity['start']}:{entity['end']}"
#     )
#     print("-" * 40)

def merge_entities(text: str, entities: list[dict]) -> list[dict]:
    if not entities:
        return []

    merged = []

    current_start = entities[0]["start"]
    current_end = entities[0]["end"]
    scores = [float(entities[0]["score"])]


    def get_sentence_context(
        text: str,
        start: int,
        end: int
    ) -> str:

        # Replace line breaks inside sentences with spaces
        normalized = re.sub(r"\s*\n\s*", " ", text)

        # Find the actual skill text first
        skill_text = text[start:end].strip()

        # Locate that same text in the normalized version
        normalized_start = normalized.find(skill_text)

        if normalized_start == -1:
            return skill_text

        normalized_end = normalized_start + len(skill_text)

        sentence_pattern = re.compile(
            r'[^.!?]+[.!?]?'
        )

        for match in sentence_pattern.finditer(normalized):

            if (
                normalized_start >= match.start()
                and normalized_end <= match.end()
            ):
                return match.group().strip()

        return skill_text
    
    for entity in entities[1:]:
        start = entity["start"]
        end = entity["end"]

        gap = text[current_end:start]

        # Merge adjacent/subword pieces and entities separated
        # by only whitespace.
        if gap.strip() == "":
            current_end = end
            scores.append(float(entity["score"]))
        else:
            merged.append(
            {
                "skill": text[current_start:current_end].strip(),

                "extraction_confidence": round(
                    sum(scores) / len(scores),
                    4
                ),

                "start": current_start,
                "end": current_end,

                "context": get_sentence_context(
                    text,
                    current_start,
                    current_end
                )
            }
        )

            current_start = start
            current_end = end
            scores = [float(entity["score"])]

    merged.append(
    {
        "skill": text[current_start:current_end].strip(),

        "extraction_confidence": round(
            sum(scores) / len(scores),
            4
        ),

        "start": current_start,
        "end": current_end,

        "context": get_sentence_context(
            text,
            current_start,
            current_end
        )
    }
    )
    
    return merged

raw_entities = extractor(job_description)

skills = merge_entities(
    job_description,
    raw_entities
)

classified_skills = classify_skills(skills)

for skill in skills:
    print(f"Skill: {skill['skill']}")
    print(f"Confidence: {skill['extraction_confidence']:.4f}")
    print(f"Context: {skill['context']}")
    print("-" * 50)

for skill in classified_skills:
    print(f"Skill: {skill['skill']}")
    print(
        f"Extraction confidence: "
        f"{skill['extraction_confidence']:.4f}"
    )
    print(f"Context: {skill['context']}")
    print(
        f"Importance: "
        f"{skill['importance_category']}"
    )
    print(
        f"Rule score: "
        f"{skill['rule_score']}"
    )
    print(
        f"Matched rule: "
        f"{skill['matched_rule']}"
    )
    print("-" * 60)