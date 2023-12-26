from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
from diffusers.loaders import LoraLoaderMixin
from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, DPMSolverSDEScheduler
import torch
import time
from compel import Compel
from xformers.ops import MemoryEfficientAttentionCutlassOp


def gen_image_float16(prompt: str,
                      cfg_scale: int = 5,
                      steps: int = 35,
                      height: int = 512):
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path="./aingdif/",
        use_safetensors=True,
        torch_dtype=torch.float16,
        load_safety_checker=False,
        extract_ema=True,
    ).to("cuda")

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config)
    pipeline.load_textual_inversion("./etc/easynegative.safetensors",
                                    token="easynegative")
    pipeline.load_textual_inversion("./etc/easynegative_2.safetensors",
                                    token="easynegative2")

    LoraLoaderMixin.load_lora_weights(
        pipeline,
        pretrained_model_name_or_path_or_dict=
        "./etc/beautifulDetailedEyes_v10.safetensors",
        adapter_name="beautifulEyes",
    )
    pipeline.enable_xformers_memory_efficient_attention(
        attention_op=MemoryEfficientAttentionCutlassOp)
    compel_proc = Compel(tokenizer=pipeline.tokenizer,
                         text_encoder=pipeline.text_encoder)

    prompt = "beautifulEyes" + prompt

    tm = time.strftime("%Y%m%d_%H%M%S")

    prompt_embeds = compel_proc(prompt)

    image = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt=
        "easynegative, easynegative2, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        num_inference_steps=steps,
        clip_skip=2,
        guidance_scale=cfg_scale,
        height=height,
        width=512,
    ).images[0]

    return image


def gen_image_float32(prompt: str,
                      cfg_scale: int = 5,
                      steps: int = 35,
                      height: int = 512):
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path="./aingdif/",
        use_safetensors=True,
        torch_dtype=torch.float32,
        load_safety_checker=False,
        extract_ema=True,
    ).to("cuda")

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config)
    pipeline.load_textual_inversion("./etc/easynegative.safetensors",
                                    token="easynegative")
    pipeline.load_textual_inversion("./etc/easynegative_2.safetensors",
                                    token="easynegative2")

    LoraLoaderMixin.load_lora_weights(
        pipeline,
        pretrained_model_name_or_path_or_dict=
        "./etc/beautifulDetailedEyes_v10.safetensors",
        adapter_name="beautifulEyes",
    )
    pipeline.enable_xformers_memory_efficient_attention(
        attention_op=MemoryEfficientAttentionCutlassOp)
    compel_proc = Compel(tokenizer=pipeline.tokenizer,
                         text_encoder=pipeline.text_encoder)

    prompt = "beautifulEyes" + prompt

    tm = time.strftime("%Y%m%d_%H%M%S")

    prompt_embeds = compel_proc(prompt)

    image = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt=
        "easynegative, easynegative2, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        num_inference_steps=steps,
        clip_skip=2,
        guidance_scale=cfg_scale,
        height=height,
        width=512,
    ).images[0]

    return image
