from app.classifiers.semantic_relevance import (
    SemanticRelevanceScorer
)


scorer = SemanticRelevanceScorer()


tests = [
    (
        "Azure Databricks",
        "You will build scalable data pipelines using "
        "Azure Databricks, PySpark and Delta Lake."
    ),
    (
        "PySpark",
        "You will build scalable data pipelines using "
        "Azure Databricks, PySpark and Delta Lake."
    ),
    (
        "Delta Lake",
        "You will build scalable data pipelines using "
        "Azure Databricks, PySpark and Delta Lake."
    ),
]

tests1 = [
    (
        "Azure Databricks",
        "You will build scalable data pipelines using "
        "Azure Databricks, PySpark and Delta Lake."
    ),
    (
        "PySpark",
        "You will build scalable data pipelines using "
        "Azure Databricks, PySpark and Delta Lake."
    ),
    (
        "Delta Lake",
        "You will build scalable data pipelines using "
        "Azure Databricks, PySpark and Delta Lake."
    ),
    (
        "Tableau",
        "You will build scalable data pipelines using "
        "Azure Databricks, PySpark and Delta Lake."
    ),
    (
        "Java",
        "You will build scalable data pipelines using "
        "Azure Databricks, PySpark and Delta Lake."
    ),
]


for skill, context in tests1:
    score = scorer.score(
        skill=skill,
        context=context
    )

    print(f"{skill:<25} score={score:.4f}")