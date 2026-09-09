from app.providers.ollama_provider import OllamaProvider
from app.services.job_extractor import JobExtractor


llm = OllamaProvider()

extractor = JobExtractor(llm)


job_description = """
JOB TITLE: Senior Data Engineer

SALARY: £72,702 - £80,780

LOCATION: Bristol

We are looking for an experienced Senior Data Engineer.

You will build scalable data pipelines using Azure Databricks,
PySpark and Delta Lake.

Strong SQL experience is essential.

Experience with CI/CD and Azure DevOps is required.

Experience with Apache Airflow would be desirable.

Candidates must have at least 4 years of data engineering
experience.

The role follows a hybrid working model with two days per
week in the Bristol office.
"""


job = extractor.extract(job_description)


print(job.model_dump_json(indent=2))