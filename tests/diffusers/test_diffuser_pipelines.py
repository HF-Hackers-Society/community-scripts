import pytest
from hf_community_scripts.diffusers.loaders.sdxl_loader import load_pipeline


def test_sdxl():
	load_pipeline()


def test_sdxl_freeu():
	with pytest.raises(Exception) as e_info:
		load_pipeline(
			do_freeu = {
				's3': 1.0
			}
		)
