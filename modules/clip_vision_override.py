from comfy.clip_vision import clip_preprocess
from comfy.ldm.modules.attention import optimized_attention_for_device
import comfy.model_management


def encoder(encoder, x, mask=None):
    optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)
    intermediates = []
    for i, l in enumerate(encoder.layers):
        x = l(x, mask, optimized_attention)
        intermediates.append(x.clone())
    return intermediates


def vision_model(vision_model, pixel_values):
    x = vision_model.embeddings(pixel_values)
    x = vision_model.pre_layrnorm(x)
    intermediates = encoder(vision_model.encoder, x)
    return intermediates


def encode_image(clip_vision, image):
    comfy.model_management.load_model_gpu(clip_vision.patcher)
    pixel_values = clip_preprocess(image.to(clip_vision.load_device), size=clip_vision.image_size).float()
    intermediates = vision_model(clip_vision.model.vision_model, pixel_values)
    return intermediates
