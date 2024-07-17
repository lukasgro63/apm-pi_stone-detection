from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ml_model import MLModel


@pytest.fixture
def mock_tflite_interpreter():
    with patch('tflite_runtime.interpreter.Interpreter') as mock:
        mock.return_value.allocate_tensors = MagicMock()
        mock.return_value.get_input_details = MagicMock(return_value=[{'shape': [1, 224, 224, 3], 'index': 0}])
        mock.return_value.get_output_details = MagicMock(return_value=[{'index': 0}])
        mock.return_value.set_tensor = MagicMock()
        mock.return_value.invoke = MagicMock()
        mock.return_value.get_tensor = MagicMock(return_value=[0.9])
        return mock

@pytest.fixture
def ml_model(mock_tflite_interpreter):
    return MLModel("dummy_model_path.tflite")

@pytest.mark.asyncio
async def test_load_model_async(ml_model, mock_tflite_interpreter):
    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock()
        await ml_model.load_model_async()
        mock_tflite_interpreter.assert_called_once_with(model_path="dummy_model_path.tflite")
        assert ml_model.interpreter is not None

def test_predict_object_success(ml_model, mock_tflite_interpreter):
    correct_image = np.zeros((1, 224, 224, 3), dtype=np.float32)
    prediction = ml_model.predict_object(correct_image)
    mock_tflite_interpreter.return_value.set_tensor.assert_called_once()
    mock_tflite_interpreter.return_value.invoke.assert_called_once()
    mock_tflite_interpreter.return_value.get_tensor.assert_called_once()
    assert prediction == 0.9, "Expected a successful prediction return value"

def test_predict_object_failure(ml_model, mock_tflite_interpreter):
    incorrect_image = np.zeros((224, 224, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        ml_model.predict_object(incorrect_image)

@pytest.mark.asyncio
async def test_load_model_failure(ml_model):
    with patch('asyncio.get_event_loop') as mock_loop, \
         patch.object(ml_model, '_load_model', side_effect=Exception("Test Error")):
        mock_loop.return_value.run_in_executor = AsyncMock(side_effect=Exception("Test Error"))
        await ml_model.load_model_async()
        assert ml_model.interpreter is None
