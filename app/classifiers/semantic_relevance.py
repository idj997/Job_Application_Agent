from sentence_transformers import CrossEncoder


class SemanticRelevanceScorer:

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def __init__(self):
        self.model = CrossEncoder(self.MODEL_NAME)

    def score(
        self,
        skill: str,
        context: str
    ) -> float:

        score = self.model.predict(
            [(skill, context)]
        )[0]

        return float(score)