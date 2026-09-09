from app.classifiers.requirement_nli import (
    RequirementNLIClassifier
)


classifier = RequirementNLIClassifier()


tests = [
    {
        "skill": "Azure Databricks",
        "context": (
            "You will build scalable data pipelines using "
            "Azure Databricks, PySpark and Delta Lake."
        ),
    },

    {
        "skill": "PySpark",
        "context": (
            "You will build scalable data pipelines using "
            "Azure Databricks, PySpark and Delta Lake."
        ),
    },

    {
        "skill": "Delta Lake",
        "context": (
            "You will build scalable data pipelines using "
            "Azure Databricks, PySpark and Delta Lake."
        ),
    },

    {
        "skill": "SQL",
        "context": (
            "Strong SQL experience is essential."
        ),
    },

    {
        "skill": "Apache Airflow",
        "context": (
            "Experience with Apache Airflow would be desirable."
        ),
    },
]


for item in tests:

    skill = item["skill"]
    context = item["context"]

    hypothesis = (
        f"{skill} is a core requirement for this role."
    )

    result = classifier.classify(
        context=context,
        hypothesis=hypothesis
    )

    print()
    print("Skill:", skill)
    print("Context:", context)
    print("Hypothesis:", hypothesis)

    print("Prediction:", result["label"])
    print(
        "Confidence:",
        round(result["confidence"], 4)
    )

    print("Probabilities:")

    for label, probability in result["probabilities"].items():
        print(
            f"  {label:<15} "
            f"{probability:.4f}"
        )

    print("-" * 60)
