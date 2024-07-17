from unittest.mock import MagicMock, patch

import pytest

from decision_unit import DecisionUnit


@pytest.fixture
def mock_config():
    return {'decision': {'threshold': 0.5}}

class TestDecisionUnit:
    def test_decision_above_threshold(self, mock_config):
        decision_unit = DecisionUnit(mock_config)
        prediction_probability = 0.6
        result = decision_unit.make_decision(prediction_probability)
        assert result == True

    def test_decision_below_threshold(self, mock_config):
        decision_unit = DecisionUnit(mock_config)
        prediction_probability = 0.4
        result = decision_unit.make_decision(prediction_probability)
        assert result == False

    def test_decision_at_threshold(self, mock_config):
        decision_unit = DecisionUnit(mock_config)
        prediction_probability = 0.5
        result = decision_unit.make_decision(prediction_probability)
        assert result == True

    @patch('decision_module.logging')
    def test_logging(self, mock_logging, mock_config):
        decision_unit = DecisionUnit(mock_config)
        prediction_probability = 0.7
        decision_unit.make_decision(prediction_probability)
        mock_logging.debug.assert_called_with(f"Decision result: True for probability {prediction_probability}")
