from onnxruntime import get_available_providers, get_device
import onnxruntime


def test_answer():
	# check available providers
	assert 'CUDAExecutionProvider' in get_available_providers(), 'ONNX Runtime GPU provider not found. Make sure onnxruntime-gpu is installed and onnxruntime is uninstalled.'
	assert 'GPU' == get_device()

	# asser version due to bug in 1.11.1
	assert onnxruntime.__version__ > '1.11.1', 'You need a newer version of ONNX Runtime!'
