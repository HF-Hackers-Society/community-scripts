import platform
import torch
from diffusers import (
	StableDiffusionXLPipeline,
	AutoencoderKL,
	UNet2DConditionModel,
	EulerDiscreteScheduler,
	EulerAncestralDiscreteScheduler,
	DPMSolverMultistepScheduler,
	DPMSolverSinglestepScheduler,
	KDPM2DiscreteScheduler,
	KDPM2AncestralDiscreteScheduler,
	UniPCMultistepScheduler,
	HeunDiscreteScheduler,
	LMSDiscreteScheduler,
)
from transformers import CLIPTextModel
from safetensors.torch import load_file as load_tensors
from ...utils import flush, log_gpu_cache
from typing import List, Dict
from huggingface_hub import cached_download
from transformers import (
	CLIPTokenizer,
	CLIPTextModelWithProjection
)


# https://huggingface.co/docs/diffusers/v0.25.0/en/using-diffusers/sdxl#optimizations
HAS_LINUX = platform.system().lower() == 'linux' and torch.__version__ >= '2.0.0'
CAN_QUANT = torch.__version__ >= '2.3.0' and HAS_LINUX
GPU_MODEL = torch.cuda.get_device_properties(0).name


if CAN_QUANT:
	from torchao.quantization import (
		apply_dynamic_quant,
		change_linear_weights_to_int4_woqtensors,
		change_linear_weights_to_int8_woqtensors,
		swap_conv2d_1x1_to_linear
	)


def dynamic_quant_filter_fn(mod, *args) -> bool:
    return (
        isinstance(mod, torch.nn.Linear)
        and mod.in_features > 16
        and (mod.in_features, mod.out_features)
        not in [
            (1280, 640),
            (1920, 1280),
            (1920, 640),
            (2048, 1280),
            (2048, 2560),
            (2560, 1280),
            (256, 128),
            (2816, 1280),
            (320, 640),
            (512, 1536),
            (512, 256),
            (512, 512),
            (640, 1280),
            (640, 1920),
            (640, 320),
            (640, 5120),
            (640, 640),
            (960, 320),
            (960, 640),
        ]
    )


def conv_filter_fn(mod, *args) -> bool:
    return (
        isinstance(mod, torch.nn.Conv2d) and mod.kernel_size == (1, 1) and 128 in [mod.in_channels, mod.out_channels]
    )


def load_torch(no_bf16: bool) -> (str, torch.dtype):
	torch.cuda.set_device(0)

	if torch.cuda.is_available():
		return 'cuda', torch.float16 if no_bf16 else torch.bfloat16
	else:
		return 'cpu', torch.float32


def quantize(do_quant: str, component: UNet2DConditionModel | AutoencoderKL):
	swap_conv2d_1x1_to_linear(component, conv_filter_fn)

	if args.do_quant == 'int4weightonly':
		change_linear_weights_to_int4_woqtensors(component)
	elif args.do_quant == 'int8weightonly':
		change_linear_weights_to_int8_woqtensors(component)
	elif args.do_quant == 'int8dynamic':
		apply_dynamic_quant(component, dynamic_quant_filter_fn)
	else:
		raise ValueError(f'Unknown do_quant value: {do_quant}.')

	torch._inductor.config.force_fuse_int_mm_with_mul = True
	torch._inductor.config.use_mixed_mm = True


# https://huggingface.co/docs/diffusers/v0.25.0/en/api/schedulers/overview#schedulers
def get_scheduler(model_args: Dict[str, str], scheduler_id: str):
	if scheduler_id == 'euler':
		return EulerDiscreteScheduler.from_pretrained(**model_args)
	elif scheduler_id == 'euler_a':
		return EulerAncestralDiscreteScheduler.from_pretrained(**model_args)
	elif scheduler_id == 'dpmpp_2m_sde_karras':
		return DPMSolverMultistepScheduler.from_pretrained(
			use_karras_sigmas=True,
			algorithm_type='sde-dpmsolver++',
			**model_args
		)
	elif scheduler_id == 'dpmpp_2m_sde':
		return DPMSolverMultistepScheduler.from_pretrained(
			algorithm_type='sde-dpmsolver++',
			**model_args
		)
	elif scheduler_id == 'dpmpp_2m_karras':
		return DPMSolverMultistepScheduler.from_pretrained(
			use_karras_sigmas=True,
			**model_args
		)
	elif scheduler_id == 'dpmpp_2m':
		return DPMSolverMultistepScheduler.from_pretrained(**model_args)
	elif scheduler_id == 'dpmpp_sde':
		return DPMSolverSinglestepScheduler.from_pretrained(**model_args)
	elif scheduler_id == 'dpmpp_sde_karras':
		return DPMSolverSinglestepScheduler.from_pretrained(
			use_karras_sigmas=True,
			**model_args
		)
	elif scheduler_id == 'huen':
		return HeunDiscreteScheduler.from_pretrained(**model_args)
	elif scheduler_id == 'lms':
		return LMSDiscreteScheduler.from_pretrained(**model_args)
	elif scheduler_id == 'lms_karras':
		return LMSDiscreteScheduler.from_pretrained(
			use_karras_sigmas=True,
			**model_args
		)
	elif scheduler_id == 'unipc':
		return UniPCMultistepScheduler.from_pretrained(**model_args)
	elif scheduler_id == 'dpm2':
		return KDPM2DiscreteScheduler.from_pretrained(**model_args)
	elif scheduler_id == 'dpm2_karras':
		return KDPM2DiscreteScheduler.from_pretrained(
			use_karras_sigmas=True,
			**model_args
		)
	elif scheduler_id == 'dpm2_a':
		return KDPM2AncestralDiscreteScheduler.from_pretrained(**model_args)
	elif scheduler_id == 'dpm2_a_karras':
		return KDPM2AncestralDiscreteScheduler.from_pretrained(
			use_karras_sigmas=True,
			**model_args
		)

	raise ValueError(f'Unknown Scheduler ID: "{scheduler_id}"')


"""
load_single_gpu_pipeline

Sends a model to a device automatically based on whether the required
GPU technologies are available.


EXAMPLES

1) Simple init

pipe = load_single_gpu_pipeline()

2) Recommended settings

width = 1024
height = 1024
n_steps = 50

prompt = 'A majestic lion leaping from a big stone at night.'

args = {
	'use_tf32': True,
	'scheduler_id': 'dpmpp_2m_sde_karras',
	'do_tiling': width > 1024 or height > 1024,
	'do_freeu': {
		'b1': 1.1, 'b2': 1.2, 's1': 0.6, 's2': 0.4
	}
}
base = load_single_gpu_pipeline(**args)

image = base(
	prompt=prompt,
	num_inference_steps=n_steps,
	width=width,
	height=height,
).images[0]

image.save('test.png')


chkpt:
	The HuggingFace model ID or model path to use. If the former it will be
	downloaded relative to `cache_dir`. Defaults to `stabilityai/stable-diffusion-xl-base-1.0`.
cache_dir:
	Where to store any HuggingFace downloads. Defaults to `downloads/`,
	which is ignored by default in the Python gitignore template.
do_quant:
	The quantization mode to use. Its value can be one of `int4weightonly`,
	`int8weightonly`, or `int8dynamic`, defaulting to `int8dynamic` on Linux.
	If None or an empty string, quantization will be disabled. It also
	requires UNet and VAE compilation to be enabled, which is not
	supported on Windows at present.
compile_unet:
	Whether to compile the UNet. Defaults to False on Windows.
compile_vae:
	Whether to compile the UNet. Defaults to False on Windows.
compile_mode:
	Any compilation mode supported by `torch.compile`. Defaults to
	`reduce-overhead` unless quantization is enabled: then it defaults
	to `max-autotune`.
use_tf32:
	Whether to use TensorFloat32 precision. It's roughly equal to FP16 performance
	at FP32 precision. It provides free performance improvements on RTX 3K+ NVIDIA GPUs.
	On supported GPUs it's automatically enabled.
no_bf16:
	If True, FP16 is enabled. If False, BFP16 is enabled.
upcast_vae:
	Whether to upcast the VAE. If False, the numerically-stable VAE `sdxl-vae-fp16-fix`
	is used over whatever the default is due to using FP16.
fuse_projections:
	Toggles fused QKV projections for both UNet and VAE.
xformers:
	Enabled automatically if the available PyTorch version is less than 2.0.0,
	since using XFormers with those versions gives memory and inference speed
	improvements. In later PyTorch versions it only gives minor memory improvements.
	It may be more beneficial on multi-gpu setups.
prompt_embeds:
	A list of file paths to SafeTensor files with SDXL-compatible prompt embeddings.
	If there are any unique activation words they will automatically be availabe to
	the tokenizers.
scheduler_id:
	The scheduler or algorithm configuration used to generate an image. The SDXL default
	is `euler`, but it can also be one of the following values:
	- `euler_a`
	- `dpmpp_2m_sde_karras` (RECOMMENDED)
	- `dpmpp_2m_sde`
	- `dpmpp_2m_karras`
	- `dpmpp_2m`
	- `dpmpp_sde`
	- `dpmpp_sde_karras`
	- `huen`
	- `lms`
	- `lms_karras`
	- `unipc` (solid option)
	- `dpm2`
	- `dpm2_karras`
	- `dpm2_a`
	- `dpm2_a_karras`
do_freeu:
	Fixes image deformations and can improve prompt adherence. If None, no changes
	are made. Otherwise pass a dictionary with str keys being any of `b1`, `b2`, `s1`,
	or `s2`, and values being the float value to set them to. If the dictionary is
	present and any of the four values is absent, it will be substituted with a default
	value that creates no image generation changes. These values are b1=1.0, b2=1.2,
	s1=0.0, and s2=0.0.

	Recommended by ChenyangSi:
	b1=1.3, b2=1.4, s1=0.9, s2=0.2

	Recommended by nasirk24 & T145:
	b1=1.1, b2=1.2, s1=0.6, s2=0.4

	References:
	- https://github.com/ChenyangSi/FreeU/tree/main#sdxl
	- https://wandb.ai/nasirk24/UNET-FreeU-SDXL/reports/FreeU-SDXL-Optimal-Parameters--Vmlldzo1NDg4NTUw?accessToken=6745kr9rjd6e9yjevkr9bpd2lm6dpn6j00428gz5l60jrhl3gj4gubrz4aepupda
do_tiling:
	Toggles VAE tiling, which improves performance when generating highres images beyond
	the default 1024x1024 resolution.
"""
def load_single_gpu_pipeline(
	chkpt: str = 'stabilityai/stable-diffusion-xl-base-1.0',
	cache_dir: str = 'downloads/',
	do_quant: str = 'int8dynamic' if CAN_QUANT else None,
	compile_unet: bool = HAS_LINUX,
	compile_vae: bool = HAS_LINUX,
	compile_mode: str = 'max-autotune' if CAN_QUANT else 'reduce-overhead',
	use_tf32: bool = 'RTX' in GPU_MODEL and '20' not in GPU_MODEL,
	no_bf16: bool = True,
	upcast_vae: bool = True,
	fuse_projections: bool = True,
	xformers: bool = torch.__version__ < '2.0.0', # If using PyTorch 2+, this only saves about ~0.5 GB!
	prompt_embeds: List[str] = list(),
	scheduler_id: str = 'euler',
	do_freeu: Dict[str, float] = None,
	do_tiling: bool = False,
	nn_benchmark: bool = False,
) -> StableDiffusionXLPipeline:
	if do_quant and not compile_unet:
		raise ValueError('Compilation for UNet must be enabled when quantizing.')
	if do_quant and not compile_vae:
		raise ValueError('Compilation for VAE must be enabled when quantizing.')

	flush()

	# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
	torch.backends.cudnn.benchmark = nn_benchmark

	if use_tf32:
		# https://huggingface.co/docs/diffusers/optimization/fp16#use-tensorfloat32
		# https://huggingface.co/docs/transformers/en/perf_train_gpu_one#tf32
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

	device, dtype = load_torch(no_bf16)
	print(f'Using dtype: {dtype}')

	uni_args = {
		'cache_dir': cache_dir,
		'torch_dtype': dtype,
	}
	model_args = {
		'pretrained_model_name_or_path': chkpt,
		'add_watermarker': False,
		**uni_args
	}

	pipeline = cached_download(
		url='https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/lpw_stable_diffusion_xl.py',
		cache_dir=cache_dir,
		force_filename='lpw_stable_diffusion_xl.py'
	)

	# 'clip-vit-large-patch14' is older!
	text_encoder = CLIPTextModel.from_pretrained(
		'openai/clip-vit-large-patch14-336',
		**uni_args
	)

	text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
		'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
		**uni_args
	)

	tokenizer_2 = CLIPTokenizer.from_pretrained(
		'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
		**uni_args
	)

	scheduler = get_scheduler({
		'pretrained_model_name_or_path': chkpt,
		'cache_dir': cache_dir,
		'subfolder': 'scheduler',
	}, scheduler_id)

	if dtype == torch.float16:
		model_args['variant'] = 'fp16'

	pipe = StableDiffusionXLPipeline.from_pretrained(
		scheduler=scheduler,
		text_encoder=text_encoder,
		text_encoder_2=text_encoder_2,
		tokenizer_2=tokenizer_2,
		use_safetensors=True,
		custom_pipeline=pipeline,
		**model_args
	)

	if do_freeu:
		freeu_defaults = {
			'b1': 1.0, 'b2': 1.2, 's1': 0.0, 's2': 0.0
		}

		for i in freeu_defaults.keys():
			if i not in do_freeu.values():
				do_freeu[i] = freeu_defaults[i]

		if len(do_freeu) > 4:
			raise ValueError('"do_freeu" can only have up to four entries: b1, b2, s1, s2')

		pipe.enable_freeu(
			b1=do_freeu['b1'],
			b2=do_freeu['b2'],
			s1=do_freeu['s1'],
			s2=do_freeu['s2']
		)

	for tensors in prompt_embeds:
		tensors = path.normpath(tensors)
		state_dict = load_tensors(tensors)
		token = path.splitext(path.basename(tensors))[0]
		pipe.load_textual_inversion(state_dict['clip_g'], token, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
		pipe.load_textual_inversion(state_dict['clip_l'], token, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

	pipe = pipe.to(device)

	# 'diffusers-fast' sends the pipe to the device after setting everything below in its runner,
	# however the docs do it before all the changes (and doing it before is significantly faster)
	# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization

	if not upcast_vae:
		print('Using a more numerically stable VAE.')
		pipe.vae = AutoencoderKL.from_pretrained(
			'madebyollin/sdxl-vae-fp16-fix', # https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
			**uni_args,
		)

	if fuse_projections:
		print('Enabling fused QKV projections for both UNet and VAE.')
		pipe.fuse_qkv_projections()

	if do_tiling:
		pipe.enable_vae_tiling()

	if upcast_vae:
		pipe.upcast_vae()

	# https://huggingface.co/docs/diffusers/main/en/optimization/memory#channels-last-memory-format
	# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-channels-last-memory-format-for-computer-vision-models
	if pipe.unet.conv_out.state_dict()['weight'].stride()[3] == 1:
		print('Flipping memory format for both UNet and VAE.')
		pipe.unet.to(memory_format=torch.channels_last)
		pipe.vae.to(memory_format=torch.channels_last)

	if compile_mode == 'max-autotune' and (compile_unet or compile_vae):
		torch._inductor.config.conv_1x1_as_mm = True
		torch._inductor.config.coordinate_descent_tuning = True
		torch._inductor.config.epilogue_fusion = False
		torch._inductor.config.coordinate_descent_check_all_directions = True

	if compile_unet:
		if do_quant:
			quantize(do_quant, pipe.unet)
			print('Applied quantization to UNet.')

		pipe.unet = torch.compile(pipe.unet, mode=compile_mode, fullgraph=True)
		print('Compiled UNet.')

	if compile_vae:
		if do_quant:
			quantize(do_quant, pipe.vae)
			print('Applied quantization to VAE.')

		pipe.vae.decode = torch.compile(pipe.vae.decode, mode=compile_mode, fullgraph=True)
		print('Compiled VAE.')

	if xformers:
		pipe.enable_xformers_memory_efficient_attention()

	# torch.cuda.synchronize()

	if device == 'cuda':
		log_gpu_cache()

	#pipe.set_progress_bar_config(disable=True)
	return pipe
