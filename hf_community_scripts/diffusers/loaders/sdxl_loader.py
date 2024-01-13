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
from typing import List


HAS_LINUX = platform.system().lower() == 'linux'
CAN_QUANT = torch.__version__ >= '2.3.0'

if CAN_QUANT:
	from torchao.quantization import (
		apply_dynamic_quant,
		change_linear_weights_to_int4_woqtensors,
		change_linear_weights_to_int8_woqtensors,
		swap_conv2d_1x1_to_linear
	)


def dynamic_quant_filter_fn(mod, *args):
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


def conv_filter_fn(mod, *args):
    return (
        isinstance(mod, torch.nn.Conv2d) and mod.kernel_size == (1, 1) and 128 in [mod.in_channels, mod.out_channels]
    )


def load_torch(no_bf16: bool) -> (str, torch.dtype):
	if torch.cuda.is_available():
		return 'cuda', torch.float16 if no_bf16 else torch.bfloat16
	else:
		return 'cpu', torch.float32


def quantize(do_quant: str, component: UNet2DConditionModel | AutoencoderKL):
	swap_conv2d_1x1_to_linear(component, conv_filter_fn)

	if args.do_quant == "int4weightonly":
		change_linear_weights_to_int4_woqtensors(component)
	elif args.do_quant == "int8weightonly":
		change_linear_weights_to_int8_woqtensors(component)
	elif args.do_quant == "int8dynamic":
		apply_dynamic_quant(component, dynamic_quant_filter_fn)
	else:
		raise ValueError(f"Unknown do_quant value: {do_quant}.")

	torch._inductor.config.force_fuse_int_mm_with_mul = True
	torch._inductor.config.use_mixed_mm = True


# https://huggingface.co/docs/diffusers/v0.25.0/en/api/schedulers/overview#schedulers
def get_scheduler(model_args: dict, scheduler_id: str):
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

	raise ValueError('Unknown Scheduler ID: "{}"'.format(scheduler_id))


def load_pipeline(
	chkpt: str = 'stabilityai/stable-diffusion-xl-base-1.0',
	cache_dir: str = 'downloads/',
	do_quant: str = 'int8dynamic' if CAN_QUANT else None,
	compile_unet: bool = HAS_LINUX,
	compile_vae: bool = HAS_LINUX,
	compile_mode: str = 'max-autotune' if CAN_QUANT else 'reduce-overhead',
	use_tf32: bool = False, # Faster but slightly less accurate
	no_bf16: bool = True,
	upcast_vae: bool = True,
	fuse_projections: bool = True,
	xformers: bool = torch.__version__ < '2.0.0', # If using PyTorch 2+, this only saves about ~0.5 GB memory!
	prompt_embeds: List[str] = list(),
	scheduler_id: str = 'dpmpp_2m_sde_karras', # SDXL default is euler
) -> StableDiffusionXLPipeline:
	if do_quant and not compile_unet:
		raise ValueError("Compilation for UNet must be enabled when quantizing.")
	if do_quant and not compile_vae:
		raise ValueError("Compilation for VAE must be enabled when quantizing.")

	flush()

	if use_tf32:
		# https://huggingface.co/docs/diffusers/optimization/fp16#use-tensorfloat32
		# https://huggingface.co/docs/transformers/en/perf_train_gpu_one#tf32
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

	device, dtype = load_torch(no_bf16)
	print(f"Using dtype: {dtype}")

	uni_args = {
		'cache_dir': cache_dir,
		'torch_dtype': dtype,
	}
	model_args = {
		'pretrained_model_name_or_path': chkpt,
		'add_watermarker': False,
		**uni_args
	}

	# pipeline = cached_download(
	# 	url='https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/lpw_stable_diffusion_xl.py',
	# 	cache_dir=model_args['cache_dir'],
	# 	force_filename='lpw_stable_diffusion_xl.py'
	# )

	# "clip-vit-large-patch14" is older!
	text_encoder = CLIPTextModel.from_pretrained(
		'openai/clip-vit-large-patch14-336',
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
		use_safetensors=True,
		**model_args
	)

	for tensors in prompt_embeds:
		tensors = path.normpath(tensors)
		state_dict = load_tensors(tensors)
		token = path.splitext(path.basename(tensors))[0]
		pipe.load_textual_inversion(state_dict["clip_g"], token, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
		pipe.load_textual_inversion(state_dict["clip_l"], token, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

	pipe = pipe.to(device)

	# "diffusers-fast" sends the pipe to the device after setting everything below in its runner,
	# however the docs do it before all the changes (and doing it before is significantly faster)

	if not upcast_vae:
		print("Using a more numerically stable VAE.")
		pipe.vae = AutoencoderKL.from_pretrained(
			'madebyollin/sdxl-vae-fp16-fix', # https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
			**uni_args,
		)

	if fuse_projections:
		print("Enabling fused QKV projections for both UNet and VAE.")
		pipe.fuse_qkv_projections()

	if upcast_vae:
		pipe.upcast_vae()

	# https://huggingface.co/docs/diffusers/main/en/optimization/memory#channels-last-memory-format
	if pipe.unet.conv_out.state_dict()["weight"].stride()[3] == 1:
		print("Flipping memory format for both UNet and VAE.")
		pipe.unet.to(memory_format=torch.channels_last)
		pipe.vae.to(memory_format=torch.channels_last)

	if compile_mode == "max-autotune" and (compile_unet or compile_vae):
		torch._inductor.config.conv_1x1_as_mm = True
		torch._inductor.config.coordinate_descent_tuning = True
		torch._inductor.config.epilogue_fusion = False
		torch._inductor.config.coordinate_descent_check_all_directions = True

	if compile_unet:
		if do_quant:
			quantize(do_quant, pipe.unet)
			print("Applied quantization to UNet.")

		pipe.unet = torch.compile(pipe.unet, mode=compile_mode, fullgraph=True)
		print("Compiled UNet.")

	if compile_vae:
		if do_quant:
			quantize(do_quant, pipe.vae)
			print("Applied quantization to VAE.")

		pipe.vae.decode = torch.compile(pipe.vae.decode, mode=compile_mode, fullgraph=True)
		print("Compiled VAE.")

	if xformers:
		pipe.enable_xformers_memory_efficient_attention()

	if device == 'cuda':
		log_gpu_cache()

	pipe.set_progress_bar_config(disable=True)
	return pipe
