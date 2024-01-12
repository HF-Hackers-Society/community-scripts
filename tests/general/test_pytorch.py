import torch


def test_answer():
	assert torch.cuda.is_available()
