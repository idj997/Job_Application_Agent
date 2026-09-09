from app.job_parser import JobDescription


class JobExtractor:

    def __init__(self, llm):
        self.llm = llm

    def extract(self, job_text: str) -> JobDescription:

        prompt = f"""
You are a structured job-description extraction system.

Your task is NOT to summarise the job description.

Your task is to extract normalized structured data.

Use ONLY information explicitly present in the job description.

==================================================
SKILL EXTRACTION RULES
==================================================

A skill must be returned as a SHORT CANONICAL NAME.

CORRECT:
"SQL"
"PySpark"
"Azure Databricks"
"Delta Lake"
"Azure DevOps"
"CI/CD"
"Apache Airflow"

INCORRECT:
"Strong SQL experience"
"Experience with Azure Databricks"
"Knowledge of PySpark"
"At least 4 years of data engineering experience"
"Build scalable data pipelines"

Never put complete sentences inside required_skills or preferred_skills.

==================================================
REQUIRED VS PREFERRED
==================================================

Treat a technical skill as REQUIRED when:

- explicitly called required
- explicitly called essential
- described as a must-have
- strong experience is requested
- the candidate is expected to use that technology
  as part of the core responsibilities

IMPORTANT:

Technologies explicitly stated as part of core job responsibilities
should normally be considered required unless the job description
indicates otherwise.

Example:

"You will build pipelines using Databricks, PySpark and Delta Lake."

means:

required_skills:
- Azure Databricks
- PySpark
- Delta Lake

Treat skills as PREFERRED when wording includes:

- desirable
- preferred
- advantageous
- nice to have
- bonus

==================================================
EXPERIENCE RULES
==================================================

Do NOT put years-of-experience statements into skills.

Example:

"Candidates must have at least 4 years of data engineering experience."

Return:

minimum_experience_years = 4
experience_area = "Data Engineering"

==================================================
LOCATION RULES
==================================================

Extract the city or geographic location when explicitly stated.

Example:

"two days per week in the Bristol office"

means:

location = "Bristol"
work_pattern = "hybrid"
office_days_per_week = 2

==================================================
SALARY RULES
==================================================

Convert salary amounts to integers.

Example:

£72,702 - £80,780

means:

salary_min = 72702
salary_max = 80780
currency = "GBP"

==================================================
SPONSORSHIP RULES
==================================================

Never infer sponsorship from:
- company size
- industry
- location
- reputation

If sponsorship is not explicitly discussed:

sponsorship = "unknown"

==================================================
SECURITY CLEARANCE RULES
==================================================

If security clearance is not mentioned:

security_clearance = "not_mentioned"

==================================================

JOB DESCRIPTION:

{job_text}
"""

        system_prompt = """
            You are an information extraction engine.

            Return structured facts rather than prose.

            Be conservative about facts that are not explicitly stated,
            but normalize technical skill names.

            Never return full requirement sentences as skill names.
            """

        return self.llm.generate_structured(
            prompt=prompt,
            schema=JobDescription,
            system_prompt=system_prompt
        )