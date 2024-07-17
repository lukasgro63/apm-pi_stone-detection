import logging

class DecisionUnit:
    def __init__(self, config):
        self.threshold = config['decision']['threshold']

    def make_decision(self, prediction_probability):
        result = prediction_probability >= self.threshold
        logging.debug(f"Decision result: {result} for probability {prediction_probability}")
        return result

