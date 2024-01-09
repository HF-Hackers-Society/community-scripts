import platform
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from ...utils import flush, log_gpu_cache


HAS_LINUX = platform.system().lower() == 'linux'


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
	if torch.__version__ < '3.0.0':
		return

	from torchao.quantization import (
		apply_dynamic_quant,
		change_linear_weights_to_int4_woqtensors,
		change_linear_weights_to_int8_woqtensors,
		swap_conv2d_1x1_to_linear,
	)

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


def load_pipeline(
	chkpt: str = 'stabilityai/stable-diffusion-xl-base-1.0',
	cache_dir: str = 'downloads/',
	do_quant: str = 'int8dynamic' if HAS_LINUX else None,
	compile_unet: bool = HAS_LINUX,
	compile_vae: bool = HAS_LINUX,
	compile_mode: str = 'max-autotune',
	use_tf32: bool = False, # Faster but slightly less accurate
	no_bf16: bool = True,
	upcast_vae: bool = True,
	fuse_projections: bool = True,
	xformers: bool = torch.__version__ < '2.0.0', # If using PyTorch 2+, this only saves about ~0.5 GB memory!
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

	if dtype == torch.float16:
		model_args['variant'] = 'fp16'

	pipe = StableDiffusionXLPipeline.from_pretrained(
		use_safetensors=True,
		**model_args
	).to(device)

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
		swap_conv2d_1x1_to_linear(pipe.unet, conv_filter_fn)

		if do_quant:
			quantize(do_quant, pipe.unet)
			print("Applied quantization to UNet.")

		pipe.unet = torch.compile(pipe.unet, mode=compile_mode, fullgraph=True)
		print("Compiled UNet.")

	if compile_vae:
		swap_conv2d_1x1_to_linear(pipe.vae, conv_filter_fn)

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
