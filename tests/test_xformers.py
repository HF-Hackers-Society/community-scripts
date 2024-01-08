import xformers


def test_answer():
	assert xformers.__version__ >= '0.0.23.post1'
