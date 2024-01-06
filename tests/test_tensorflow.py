import tensorflow as tf


def test_answer():
	assert tf.config.list_physical_devices('GPU')
