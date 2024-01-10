import torch
import math


def flush():
	torch.cuda.empty_cache()

	if torch.__version__ < '2.3.0':
		torch.cuda.reset_max_memory_allocated()

	torch.cuda.reset_peak_memory_stats()


# https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
def convert_size(size_bytes):
	if size_bytes == 0:
		return '0B'
	suffix = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
	i = int(math.floor(math.log(size_bytes, 1024)))
	p = math.pow(1024, i)
	s = round(size_bytes / p, 2)
	return '%s %s' % (s, suffix[i])


def log_gpu_cache():
	print('CUDA Memory Allocated: {}'.format(convert_size(torch.cuda.memory_allocated(0))))
	print('CUDA Memory Reserved: {}'.format(convert_size(torch.cuda.memory_reserved(0))))
	print('CUDA Max Memory Allocated: {}'.format(convert_size(torch.cuda.max_memory_reserved(0))))
