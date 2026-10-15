"""
Lunar Explorer AI
-----------------
Experimental hybrid predictor.

Allows independent selection of the prior and
observation models.

Hybrid configurations are diagnostic experiments:
the two models may predict different target events.
"""


class HybridIcePredictor:

    def __init__(self, prior_predictor, observation_predictor):

        self.prior_predictor = prior_predictor
        self.observation_predictor = observation_predictor

    def predict_prior_map(self, environment):

        return self.prior_predictor.predict_prior_map(
            environment
        )

    def predict_observation(
        self,
        temperature,
        slope,
        neutron,
        gpr,
    ):

        return self.observation_predictor.predict_observation(
            temperature,
            slope,
            neutron,
            gpr,
        )
