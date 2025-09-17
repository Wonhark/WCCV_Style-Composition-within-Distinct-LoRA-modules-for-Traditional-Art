import os
import argparse
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

import sys
sys.path.append('/workspace/diffusers/src')
from diffusers.utils import BaseOutput
from dataclasses import dataclass

from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers import (
    FluxControlNetPipeline,
    FluxControlNetModel,
    FluxMultiControlNetModel,
    FluxPipeline
)
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import load_image
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import random
import math

# import wandb

from tqdm import tqdm


@dataclass
class UniInvEulerSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


@dataclass
class UniInvDDIMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_tensor(encoder_hidden_states: torch.Tensor,
               timestep,
               save_root: str = "/home/data/jaehyun/wccv"
                ):
    """
    hidden_states: (batch_size, seq_len, hidden_dim) 형태의 텐서
    timestep: 현재 diffusion timestep (정수)
    save_dir: 저장할 디렉토리 경로
    """
    os.makedirs(save_root, exist_ok=True)
    filepath = os.path.join(save_root, f"latents_{int(timestep)}.pt")
    torch.save(encoder_hidden_states.detach().cpu(), filepath)


def load_tensor(timestep,
                save_root: str = "/home/data/jaehyun/wccv/latents",
                device: str = "cuda") -> torch.Tensor:
    """
    timestep: 저장할 때 사용한 diffusion step (정수)
    save_root: 저장 디렉토리
    device: 'cpu' 또는 'cuda'로 로드할 디바이스 지정
    """
    # 파일 경로 구성
    filepath = os.path.join(save_root, f"latents_{int(timestep)}.pt")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")

    # 텐서 로드 (map_location으로 바로 원하는 디바이스에 올릴 수 있음)
    tensor = torch.load(filepath, map_location=device)
    return tensor


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def retrieve_inv_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_inv_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_inv_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_inv_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_inv_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_inv_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_inv_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_inv_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_inv_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_inv_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def extract_depthmap(image, output_path, threshold=0.5, device='cuda'):
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)

    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
    depth = outputs.predicted_depth.squeeze().cpu().numpy()

    del feature_extractor
    del depth_model

    # 5) 정규화 → 그레이스케일 이미지 변환
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = cv2.resize(depth, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    mask1 = np.where(depth <= threshold, 1, 0)
    mask2 = np.where(depth > threshold, 1, 0)

    depth_image = Image.fromarray((depth * 255).astype(np.uint8))
    mask1_image = Image.fromarray((mask1 * 255).astype(np.uint8))
    mask2_image = Image.fromarray((mask2 * 255).astype(np.uint8))

    depth_image.save(output_path, "JPEG")
    # mask1_image.save("outputs/mask1.jpg", "JPEG")
    # mask2_image.save("outputs/mask2.jpg", "JPEG")

    # mask1 = Image.fromarray((mask1 * 255).astype(np.uint8))
    # mask2 = Image.fromarray((mask2 * 255).astype(np.uint8))
    #
    # # 6) 1024×1024로 리사이즈 후 JPG 저장
    # mask1 = mask1.resize((1024, 1024), Image.BILINEAR)
    # mask2 = mask2.resize((1024, 1024), Image.BILINEAR)

    return depth_image, torch.tensor(depth), torch.tensor(mask1), torch.tensor(mask2), mask1_image, mask2_image


class MyEulerScheduler(FlowMatchEulerDiscreteScheduler):
    zero_initial = False
    alpha = 1

    def set_hyperparameters(self, zero_initial=False, alpha=1):
        self.zero_initial = zero_initial
        self.alpha = alpha

    def set_inv_timesteps(
            self,
            num_inference_steps: int = None,
            device: Union[str, torch.device] = None,
            sigmas: Optional[List[float]] = None,
            mu: Optional[float] = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps
        else:
            self.num_inference_steps = len(sigmas)

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        # timesteps
        timesteps = sigmas * self.config.num_train_timesteps
        timesteps = torch.cat([timesteps, torch.zeros(1).to(sigmas)])
        self.timesteps = timesteps.flip(dims=[0]).to(device=device)

        # sigmas
        sigmas = torch.cat([sigmas, torch.zeros(1).to(sigmas)])
        self.sigmas = sigmas.flip(dims=[0]).to(device=device)

        # empty dt and derivative
        self.sample = None

        # zero_initial
        if self.zero_initial:
            self.timesteps = self.timesteps[1:]
            self.sigmas = self.sigmas[1:]
            self.sample = 'placeholder'
            self.first_sigma = 0

        # alpha, early stop
        if self.alpha < 1:
            inv_steps = math.floor(self.alpha * self.num_inference_steps)
            skip_steps = self.num_inference_steps - inv_steps
            self.timesteps = self.timesteps[: -skip_steps]
            self.sigmas = self.sigmas[: -skip_steps]

        self._step_index = 0
        self._begin_index = 0

    def inverse_step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            s_churn: float = 0.0,
            s_tmin: float = 0.0,
            s_tmax: float = float("inf"),
            s_noise: float = 1.0,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
    ) -> Union[UniInvEulerSchedulerOutput, Tuple]:

        if (
                isinstance(timestep, int)
                or isinstance(timestep, torch.IntTensor)
                or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `HeunDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        sample = sample.to(torch.float32)

        if self.sample is None:
            # just for the first step
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]

            derivative = model_output  # v_0 = f(t=0, x_0)
            dt = sigma_next - sigma  # sigma_{t + \Delta t} - sigma_t

            # store for correction
            self.sample = sample  # Z_0

            prev_sample = sample + derivative * dt
            prev_sample = prev_sample.to(model_output.dtype)
        else:
            sigma = self.sigmas[self.step_index - 1]
            sigma_next = self.sigmas[self.step_index]

            if isinstance(self.sample, str):
                # for zero_initial
                sigma = self.first_sigma
                self.sample = sample

            derivative = model_output
            dt = sigma_next - sigma

            sample = self.sample

            self.sample = sample + dt * derivative

            # if (self.step_index + 1) < len(self.sigmas):
            #     sigma_next_next = self.sigmas[self.step_index + 1]
            #     dt_next = sigma_next_next - sigma_next
            #
            #     prev_sample = self.sample + dt_next * derivative
            # else:
            #     # end loop
            #     prev_sample = self.sample
            prev_sample = self.sample
            prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return UniInvEulerSchedulerOutput(prev_sample=prev_sample)


class LoRAManager:
    def __init__(self, pipe):
        self.pipe = pipe
        self.loaded_loras = {}
        self.original_state = None

    def load_lora(self, lora_path, adapter_name):
        """LoRA를 메모리에 로드 (한 번만)"""
        if adapter_name not in self.loaded_loras:
            self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            self.loaded_loras[adapter_name] = lora_path
            print(f"LoRA '{adapter_name}' loaded")

    def set_active_lora(self, adapter_name=None, weight=1.0):
        """활성 LoRA 설정"""
        try:
            if adapter_name is None:
                # 모든 LoRA 비활성화 - 더 안전한 방법들
                # self._disable_all_loras()
                # self.pipe._disable_all_lora()
                self._disable_all_loras()
                print("All LoRAs disabled")
            elif adapter_name in self.loaded_loras:
                self.pipe.set_adapters([adapter_name], adapter_weights=[weight])
                self.pipe.enable_lora()
                print(f"LoRA '{adapter_name}' activated with weight {weight}")
            else:
                print(f"LoRA '{adapter_name}' not loaded")
        except Exception as e:
            print(f"Error setting LoRA: {e}")
            # fallback으로 다른 방법 시도
            self._disable_all_loras_fallback()

    def _disable_all_loras(self):
        """LoRA를 안전하게 비활성화하는 메인 방법"""
        try:
            # 방법 1: disable_lora() 사용 (가장 안전)
            if hasattr(self.pipe, 'disable_lora'):
                self.pipe.disable_lora()
                return

            # 방법 2: unload_lora_weights() 사용
            elif hasattr(self.pipe, 'unload_lora_weights'):
                self.pipe.unload_lora_weights()
                return

            # 방법 3: set_adapters with proper format
            else:
                # Flux의 경우 transformer와 text_encoder 모두 처리
                if hasattr(self.pipe, 'transformer'):
                    self.pipe.transformer.set_adapters([])
                if hasattr(self.pipe, 'text_encoder') and hasattr(self.pipe.text_encoder, 'set_adapters'):
                    self.pipe.text_encoder.set_adapters([])

        except Exception as e:
            print(f"Main disable method failed: {e}")
            raise e

    def _disable_all_loras_fallback(self):
        """fallback 방법들"""
        try:
            # 방법 4: 각 컴포넌트별로 직접 처리
            for component_name in ['transformer', 'text_encoder', 'text_encoder_2']:
                component = getattr(self.pipe, component_name, None)
                if component and hasattr(component, 'set_adapters'):
                    try:
                        component.set_adapters([])
                    except:
                        pass
        except Exception as e:
            print(f"Fallback method also failed: {e}")

    def combine_loras(self, lora_configs):
        """여러 LoRA 동시 적용"""
        adapters = []
        weights = []
        for adapter_name, weight in lora_configs:
            if adapter_name in self.loaded_loras:
                adapters.append(adapter_name)
                weights.append(weight)

        if adapters:
            try:
                self.pipe.set_adapters(adapters, adapter_weights=weights)
                print(f"Combined LoRAs: {list(zip(adapters, weights))}")
            except Exception as e:
                print(f"Error combining LoRAs: {e}")

    def get_loaded_loras(self):
        """로드된 LoRA 목록 반환"""
        return list(self.loaded_loras.keys())

    def remove_lora(self, adapter_name):
        """특정 LoRA를 완전히 제거"""
        if adapter_name in self.loaded_loras:
            try:
                self.pipe.delete_adapters([adapter_name])
                del self.loaded_loras[adapter_name]
                print(f"LoRA '{adapter_name}' removed")
            except Exception as e:
                print(f"Error removing LoRA: {e}")


class MultiFluxControlNetPipeline(nn.Module):
    def __init__(self,
                 pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
                 controlnet_checkpoint="XLabs-AI/flux-controlnet-depth-diffusers",
                 num_inference_steps=28,
                 lora_path_1=None,
                 lora_path_2=None,
                 device='cuda'
                 ):

        super().__init__()
        self.device = device

        self.controlnet = FluxControlNetModel.from_pretrained(
            controlnet_checkpoint,
            torch_dtype=torch.float16
        )

        self.pipe = FluxControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        ).to(device)

        # self.pipe.scheduler = MyEulerScheduler.from_pretrained(
        #     pretrained_model_name_or_path, subfolder="scheduler"
        # )

        self.lora_manager = LoRAManager(self.pipe)
        self.lora_manager.load_lora(lora_path_1, "style_1")
        self.lora_manager.load_lora(lora_path_2, "style_2")

        # self.prompt = prompt
        # self.prompt_1 = prompt_1
        # self.prompt_2 = prompt_2

        self.num_inference_steps = num_inference_steps

    @torch.no_grad()
    def encode_image(self, image):
        # np -> tensor
        image = torch.from_numpy(np.array(image)).to(self.pipe.dtype) / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device)
        image = torch.nn.functional.interpolate(image, (1024, 1024))
        latents = self.pipe.vae.encode(image)

        posterior = latents.latent_dist
        latents = posterior.mean  # posterior.sample()

        if self.pipe.vae.config.shift_factor is not None:
            latents = latents * self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        else:
            latents = latents * self.pipe.vae.config.scaling_factor

        if hasattr(self.pipe, '_pack_latents'):
            latents = self.pipe._pack_latents(
                latents,
                1,
                self.pipe.transformer.config.in_channels // 4,
                (int(image.shape[-2]) // self.pipe.vae_scale_factor),
                (int(image.shape[-1]) // self.pipe.vae_scale_factor)
            )
        return latents

    @torch.no_grad()
    def decode_latents(self, latents, height=1024, width=1024):
        if hasattr(self.pipe, '_unpack_latents'):
            latents = self.pipe._unpack_latents(
                latents,
                height,
                width,
                self.pipe.vae_scale_factor
            )
        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            style_prompt: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 28,
            sigmas: Optional[List[float]] = None,
            guidance_scale: float = 3.5,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            control_image = None,
            control_mode: Optional[Union[int, List[int]]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
            original_step: int = 15,
            mask = None,
            method_index=0,
    ):

        height = height or self.pipe.default_sample_size * self.pipe.vae_scale_factor
        width = width or self.pipe.default_sample_size * self.pipe.vae_scale_factor

        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(self.controlnet.nets) if isinstance(self.controlnet, FluxMultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # self.pipe.check_inputs(
        #     prompt,
        #     prompt_2,
        #     height,
        #     width,
        #     prompt_embeds=prompt_embeds,
        #     pooled_prompt_embeds=pooled_prompt_embeds,
        #     callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        #     max_sequence_length=max_sequence_length
        # )

        self._guidance_scale = guidance_scale
        self.joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        # encoding prompt
        self.lora_manager.set_active_lora(None)
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale
        )

        self.lora_manager.set_active_lora(None)
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_text_ids,
        ) = self.pipe.encode_prompt(
            prompt="",
            prompt_2="",
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale
        )

        # style 1 encoding prompt
        self.lora_manager.set_active_lora("style_1")
        (
            prompt_embeds_1,
            pooled_prompt_embeds_1,
            text_ids_1,
        ) = self.pipe.encode_prompt(
            prompt=style_prompt,
            prompt_2=style_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale
        )

        # style 2 encoding prompt
        self.lora_manager.set_active_lora("style_2")
        (
            prompt_embeds_2,
            pooled_prompt_embeds_2,
            text_ids_2,
        ) = self.pipe.encode_prompt(
            prompt=style_prompt,
            prompt_2=style_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale
        )

        # 3. Prepare control image
        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        if isinstance(self.pipe.controlnet, FluxControlNetModel):
            control_image = self.pipe.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=self.device,
                dtype=self.pipe.vae.dtype,
            )
            height, width = control_image.shape[-2:]

            controlnet_blocks_repeat = False if self.pipe.controlnet.input_hint_block is None else True
            if self.pipe.controlnet.input_hint_block is None:
                # vae encode
                control_image = retrieve_latents(self.pipe.vae.encode(control_image), generator=generator)
                control_image = (control_image - self.pipe.vae.config.shift_factor) * \
                                self.pipe.vae.config.scaling_factor

                # pack
                height_control_image, width_control_image = control_image.shape[2:]
                control_image = self._pack_latents(
                    control_image,
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height_control_image,
                    width_control_image
                )

            if control_mode is not None:
                if not isinstance(control_mode, int):
                    raise ValueError(" For `FluxControlNet`, `control_mode` should be an `int` or `None`")
                control_mode = torch.tensor(control_mode).to(self.device, dtype=torch.long)
                control_mode = control_mode.view(-1, 1).expand(control_image.shape[0], 1)

        elif isinstance(self.controlnet, FluxMultiControlNetModel):
            control_images = []
            controlnet_blocks_repeat = False if self.pipe.controlnet.nets[0].input_hint_block is None else True
            for i, control_image_ in enumerate(control_image):
                control_image_ = self.prepare_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=self.device,
                    dtype=torch.float16
                )
                height, width = control_image_.shape[-2:]

                if self.pipe.controlnet.nets[0].input_hint_block is None:
                    # vae encode
                    control_image_ = retrieve_latents(self.pipe.vae_encode(control_image_), generator=generator)
                    control_image_ = (control_image_ - self.pipe.vae.config.shift_factor) * \
                                     self.pipe.vae.config.scaling_factor

                    # pack
                    height_control_image, width_control_image = control_image.shape[2:]
                    control_image_ = self.pipe._pack_latents(
                        control_image_,
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height_control_image,
                        width_control_image,
                    )
                control_images.append(control_image_)

            control_image = control_images

            if isinstance(control_mode, list) and len(control_mode) != len(control_image):
                raise ValueError(
                    "For Multi-ControlNet, `control_mode` must be a list of the same "
                    + " length as the number of controlnets (control images) specified"
                )
            if not isinstance(control_mode, list):
                control_mode = [control_mode] * len(control_image)
            # set control mode
            control_modes = []
            for cmode in control_mode:
                if cmode is None:
                    cmode = -1
                control_mode = torch.tensor(cmode).expand(control_images[0].shape[0]).to(self.device, dtype=torch.long)
                control_modes.append(control_mode)
            control_mode = control_modes

        # 4. Prepare latent variables
        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        latents, latent_image_ids = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds_1.dtype,
            self.device,
            generator,
            latents
        )
        # prepare mask
        mask = mask.to(latents.device).to(latents.dtype)
        mask = mask.unsqueeze(0).unsqueeze(0)

        mask = F.interpolate(mask, size=(mask.shape[-2]//8, mask.shape[-1]//8), mode='bilinear', align_corners=False)
        mask = mask.repeat(1, num_channels_latents, 1, 1)
        new_height, new_width = mask.shape[-2], mask.shape[-1]
        mask = mask.view(batch_size, num_channels_latents, new_height // 2, 2, new_width // 2, 2)
        mask = mask.permute(0, 2, 4, 1, 3, 5)
        mask = mask.reshape(batch_size, (new_height // 2) * (new_width // 2), num_channels_latents * 4)
        mask = mask.to(latents.dtype)

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.pipe.scheduler.config.base_image_seq_len,
            self.pipe.scheduler.config.max_image_seq_len,
            self.pipe.scheduler.config.base_shift,
            self.pipe.scheduler.config.max_shift,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler,
            num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.pipe.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(self.pipe.controlnet, FluxControlNetModel) else keeps)

        latents_1 = latents.detach().clone()
        latents_2 = latents.detach().clone()

        # wandb_images_1 = []
        # wandb_images_2 = []
        # wandb_inter_images = []
        # wandb_images = []
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                cur_t = t
                zero_timestep = torch.tensor(0.0).to(latents.dtype).to(latents.device)
                zero_timestep = zero_timestep.expand(latents.shape[0]).to(latents.dtype)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                next_timestep = timesteps[i+1].expand(latents.shape[0]).to(latents.dtype) \
                    if i < len(timesteps) - 1 else zero_timestep

                if isinstance(self.pipe.controlnet, FluxMultiControlNetModel):
                    use_guidance = self.pipe.controlnet.nets[0].config.guidance_embeds
                else:
                    use_guidance = self.pipe.controlnet.config.guidance_embeds

                guidance = torch.tensor([guidance_scale], device=self.device) if use_guidance else None
                guidance = guidance.expand(latents.shape[0]) if guidance is not None else None

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]
                if i < original_step:

                    self.lora_manager.set_active_lora("style_1")
                    controlnet_block_samples_1_cond, controlnet_single_block_samples_1_cond = self.pipe.controlnet(
                        hidden_states=latents_1,
                        controlnet_cond=control_image,
                        controlnet_mode=control_mode,
                        conditioning_scale=cond_scale,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds_1,
                        encoder_hidden_states=prompt_embeds_1,
                        txt_ids=text_ids_1,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )

                    noise_pred_1_cond = self.pipe.transformer(
                        hidden_states=latents_1,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds_1,
                        encoder_hidden_states=prompt_embeds_1,
                        controlnet_block_samples=controlnet_block_samples_1_cond,
                        controlnet_single_block_samples=controlnet_single_block_samples_1_cond,
                        txt_ids=text_ids_1,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]

                    controlnet_block_samples_1_uncond, controlnet_single_block_samples_1_uncond = self.pipe.controlnet(
                        hidden_states=latents_1,
                        controlnet_cond=control_image,
                        controlnet_mode=control_mode,
                        conditioning_scale=cond_scale,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )

                    noise_pred_1_uncond = self.pipe.transformer(
                        hidden_states=latents_1,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        controlnet_block_samples=controlnet_block_samples_1_uncond,
                        controlnet_single_block_samples=controlnet_single_block_samples_1_uncond,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]

                    self.lora_manager.set_active_lora("style_2")
                    controlnet_block_samples_2_cond, controlnet_single_block_samples_2_cond = self.pipe.controlnet(
                        hidden_states=latents_2,
                        controlnet_cond=control_image,
                        controlnet_mode=control_mode,
                        conditioning_scale=cond_scale,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds_2,
                        encoder_hidden_states=prompt_embeds_2,
                        txt_ids=text_ids_2,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )

                    noise_pred_2_cond = self.pipe.transformer(
                        hidden_states=latents_2,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds_2,
                        encoder_hidden_states=prompt_embeds_2,
                        controlnet_block_samples=controlnet_block_samples_2_cond,
                        controlnet_single_block_samples=controlnet_single_block_samples_2_cond,
                        txt_ids=text_ids_2,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]

                    controlnet_block_samples_2_uncond, controlnet_single_block_samples_2_uncond = self.pipe.controlnet(
                        hidden_states=latents_2,
                        controlnet_cond=control_image,
                        controlnet_mode=control_mode,
                        conditioning_scale=cond_scale,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )

                    noise_pred_2_uncond = self.pipe.transformer(
                        hidden_states=latents_2,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        controlnet_block_samples=controlnet_block_samples_2_uncond,
                        controlnet_single_block_samples=controlnet_single_block_samples_2_uncond,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]

                else:
                    if i == original_step:
                        latents = mask * latents_1 + (1 - mask) * latents_2

                    self.lora_manager.set_active_lora("style_1")
                    controlnet_block_samples_1_cond, controlnet_single_block_samples_1_cond = self.pipe.controlnet(
                        hidden_states=latents,
                        controlnet_cond=control_image,
                        controlnet_mode=control_mode,
                        conditioning_scale=cond_scale,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds_1,
                        encoder_hidden_states=prompt_embeds_1,
                        txt_ids=text_ids_1,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )

                    noise_pred_1_cond = self.pipe.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds_1,
                        encoder_hidden_states=prompt_embeds_1,
                        controlnet_block_samples=controlnet_block_samples_1_cond,
                        controlnet_single_block_samples=controlnet_single_block_samples_1_cond,
                        txt_ids=text_ids_1,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]

                    controlnet_block_samples_1_uncond, controlnet_single_block_samples_1_uncond = self.pipe.controlnet(
                        hidden_states=latents,
                        controlnet_cond=control_image,
                        controlnet_mode=control_mode,
                        conditioning_scale=cond_scale,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )

                    noise_pred_1_uncond = self.pipe.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        controlnet_block_samples=controlnet_block_samples_1_uncond,
                        controlnet_single_block_samples=controlnet_single_block_samples_1_uncond,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]

                    self.lora_manager.set_active_lora("style_2")

                    controlnet_block_samples_2_cond, controlnet_single_block_samples_2_cond = self.pipe.controlnet(
                        hidden_states=latents,
                        controlnet_cond=control_image,
                        controlnet_mode=control_mode,
                        conditioning_scale=cond_scale,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds_2,
                        encoder_hidden_states=prompt_embeds_2,
                        txt_ids=text_ids_2,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )

                    noise_pred_2_cond = self.pipe.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds_2,
                        encoder_hidden_states=prompt_embeds_2,
                        controlnet_block_samples=controlnet_block_samples_2_cond,
                        controlnet_single_block_samples=controlnet_single_block_samples_2_cond,
                        txt_ids=text_ids_2,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]

                    controlnet_block_samples_2_uncond, controlnet_single_block_samples_2_uncond = self.pipe.controlnet(
                        hidden_states=latents,
                        controlnet_cond=control_image,
                        controlnet_mode=control_mode,
                        conditioning_scale=cond_scale,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )

                    noise_pred_2_uncond = self.pipe.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        controlnet_block_samples=controlnet_block_samples_2_uncond,
                        controlnet_single_block_samples=controlnet_single_block_samples_2_uncond,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]

                if i < original_step:
                    step_index = (self.pipe.scheduler.timesteps == cur_t).nonzero(as_tuple=True)[0]
                    sigma_t = self.pipe.scheduler.sigmas[step_index].to(latents.dtype)
                    sigma_next = self.pipe.scheduler.sigmas[step_index + 1].to(latents.dtype)
                    noise_pred_1 = noise_pred_1_uncond + guidance_scale * (noise_pred_1_cond - noise_pred_1_uncond)
                    noise_pred_2 = noise_pred_2_uncond + guidance_scale * (noise_pred_2_cond - noise_pred_2_uncond)
                    latents_dtype = latents_1.dtype
                    # latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    latents_1 = latents_1 + (sigma_next - sigma_t) * noise_pred_1
                    latents_2 = latents_2 + (sigma_next - sigma_t) * noise_pred_2

                    # inter_image_1 = self.decode_latents(latents_1)
                    # # inter_image_1.save(f"outputs/inter/style1_{int(t)}.png")
                    # wandb_images_1.append(wandb.Image(inter_image_1, caption=f"style1_{int(t)}"))
                    # inter_image_2 = self.decode_latents(latents_2)
                    # # inter_image_2.save(f"outputs/inter/style2_{int(t)}.png")
                    # wandb_images_2.append(wandb.Image(inter_image_2, caption=f"style2_{int(t)}"))
                else:
                    noise_pred_1 = noise_pred_1_uncond + guidance_scale * (noise_pred_1_cond - noise_pred_1_uncond)
                    noise_pred_2 = noise_pred_2_uncond + guidance_scale * (noise_pred_2_cond - noise_pred_2_uncond)
                    # noise_pred = mask * noise_pred_1 + (1 - mask) * noise_pred_2

                    # predicted x0
                    latents_dtype = latents.dtype
                    step_index = (self.pipe.scheduler.timesteps == cur_t).nonzero(as_tuple=True)[0]
                    sigma_t = self.pipe.scheduler.sigmas[step_index].to(latents.dtype)
                    sigma_next = self.pipe.scheduler.sigmas[step_index + 1].to(latents.dtype)
                    sigma_last = self.pipe.scheduler.sigmas[-1].to(latents.dtype)  # timestep 0

                    x0_pred_1 = latents + (sigma_last - sigma_t) * noise_pred_1
                    x0_pred_2 = latents + (sigma_last - sigma_t) * noise_pred_2
                    x0_combined = mask * x0_pred_1 + (1 - mask) * x0_pred_2

                    # latents_dtype = latents.dtype
                    # latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    if method_index == 0:
                        latents_1 = x0_combined - (sigma_last - sigma_t) * noise_pred_1_uncond
                        latents_2 = x0_combined - (sigma_last - sigma_t) * noise_pred_2_uncond
                    else:
                        latents_1 = x0_combined - (sigma_last - sigma_t) * noise_pred_1
                        latents_2 = x0_combined - (sigma_last - sigma_t) * noise_pred_2

                    latents_1 = latents_1 + (sigma_next - sigma_t) * noise_pred_1
                    latents_2 = latents_2 + (sigma_next - sigma_t) * noise_pred_2



                    latents = mask * latents_1 + (1 - mask) * latents_2

                    # inter_image = self.decode_latents(latents)
                    # inter_image.save(f"outputs/inter/{int(t)}.png")
                    # wandb_inter_images.append(wandb.Image(inter_image, caption=f"{int(t)}"))

                    # x0_pred_1 = latents + (sigma_last - sigma_t) * noise_pred_1
                    # x0_pred_2 = latents + (sigma_last - sigma_t) * noise_pred_2
                    # x0_combined = mask * x0_pred_1 + (1 - mask) * x0_pred_2

                    # back to t-1
                    # latents = x0_combined - (sigma_last - sigma_next) * noise_pred

                # go to x0
                # step_index = (self.pipe.scheduler.timesteps == cur_t).nonzero(as_tuple=True)[0]
                # sigma_t = self.pipe.scheduler.sigmas[step_index].to(latents.dtype)
                # sigma_next = self.pipe.scheduler.sigmas[step_index + 1].to(latents.dtype)
                # sigma_last = self.pipe.scheduler.sigmas[-1].to(latents.dtype)  # timestep 0

                # noise_pred = mask * noise_pred_1 + (1 - mask) * noise_pred_2

                # # combine at x0
                # x0_combined = mask * x0_pred_1 + (1 - mask) * x0_pred_2


                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    # prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                               (i + 1) % self.pipe.scheduler.order == 0):
                    progress_bar.update()

        # wandb.log({
        #     "style1 intermediate": wandb_images_1,
        #     "style2 intermediate": wandb_images_2,
        #     "intermediate": wandb_inter_images
        # })
        if output_type == "latent":
            image = latents
        else:
            latents = self.pipe._unpack_latents(latents, height, width, self.pipe.vae_scale_factor)
            latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor

            image = self.pipe.vae.decode(latents, return_dict=False)[0]
            image = self.pipe.image_processor.postprocess(image, output_type=output_type)

        self.pipe.maybe_free_model_hooks()

        if not return_dict:
            return (image, )

        return FluxPipelineOutput(images=image)


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, default="a dog in the mountain"
    )
    parser.add_argument(
        "--prompt1", type=str, default="a dog in the mountain, on zzv style"
    )
    parser.add_argument(
        "--prompt2", type=str, default="a dog in the mountain, on zzv style"
    )
    parser.add_argument(
        "--lora_path1", type=str, required=True
    )
    parser.add_argument(
        "--lora_path2", type=str, required=True
    )
    parser.add_argument(
        "--depth_output_path", type=str, default="outputs/single_depth.jpg"
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs/result.jpg"
    )
    parser.add_argument(
        "--latents_path", type=str, default="/home/data/jaehyun/wccv/latents"
    )
    parser.add_argument(
        "--seed", type=int, default=123
    )
    parser.add_argument(
        "--original_step", type=int, default=20
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5
    )
    parser.add_argument(
        "--project_name", type=str
    )
    parser.add_argument(
        "--output_dir", type=str
    )


    return parser


def main(args):
    seed_list = [1000*x for x in range(1,6)]
    prompt_list = [
        "a dog in the mountain",
    ]
    style_prompt_list = [
        "a dog in the mountain, zzv style",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"



    ###### original pipeline inference
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16
    ).to(device)

    for seed in seed_list:
        for prompt, style_prompt in zip(prompt_list, style_prompt_list):
            seed_everything(seed)

            dir_prompt = prompt.replace(" ", "_")
            output_dir = f"{args.output_dir}/{args.project_name}/{dir_prompt}/seed_{seed}"
            os.makedirs(f"{output_dir}", exist_ok=True)

            base_image = pipe(prompt).images[0]
            base_image.save(f"{output_dir}/base_image.png")
    del pipe
    ############################################################

    pipe = MultiFluxControlNetPipeline(
        lora_path_1=args.lora_path1,
        lora_path_2=args.lora_path2,
    )
    for seed in seed_list:
        for prompt, style_prompt in zip(prompt_list, style_prompt_list):

            dir_prompt = prompt.replace(" ", "_")
            output_dir = f"{args.output_dir}/{args.project_name}/{dir_prompt}/seed_{seed}"
            base_image_save_path = f"{output_dir}/base_image.png"
            base_image = load_image(base_image_save_path)

            depth_output_path = f"{output_dir}/depth_image.png"
            mask_image, mask, mask1, mask2, mask1_image, mask2_image = extract_depthmap(base_image, depth_output_path)
            control_image = load_image(depth_output_path)


            pipe.lora_manager.set_active_lora("style_1")
            output_1 = pipe.pipe(
                prompt=prompt,
                control_image=control_image,
                num_inference_steps=28,
                guidance_scale=3.5
            ).images[0]

            pipe.lora_manager.set_active_lora("style_2")
            output_2 = pipe.pipe(
                prompt=prompt,
                control_image=control_image,
                num_inference_steps=28,
                guidance_scale=3.5
            ).images[0]

            output_1.save(f"{output_dir}/style1.png")
            output_2.save(f"{output_dir}/style2.png")
            mask_image.save(f"{output_dir}/mask.png")
            mask1_image.save(f"{output_dir}/mask1.png")
            mask2_image.save(f"{output_dir}/mask2.png")

            for i in range(2):
                output_mix_1 = pipe(
                    prompt=prompt,
                    style_prompt=style_prompt,
                    control_image=control_image,
                    num_inference_steps=28,
                    guidance_scale=args.guidance_scale,
                    original_step=args.original_step,
                    mask=mask1,
                    method_index=i,
                ).images[0]

                output_mix_2 = pipe(
                    prompt=prompt,
                    style_prompt=style_prompt,
                    control_image=control_image,
                    num_inference_steps=28,
                    guidance_scale=args.guidance_scale,
                    original_step=args.original_step,
                    mask=mask2,
                    method_index=i,
                ).images[0]

                output_mix_1.save(f"{output_dir}/mixing1_{i}.png")
                output_mix_2.save(f"{output_dir}/mixing2_{i}.png")



if __name__ == "__main__":
    args = argparser().parse_args()
    main(args)