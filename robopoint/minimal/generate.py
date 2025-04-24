import torch
from PIL import Image
import math
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
import re
from transformers import LlamaForCausalLM

from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robopoint.conversation import conv_templates
from robopoint.model.builder import load_pretrained_model
from robopoint.utils import disable_torch_init
from robopoint.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, get_anyres_image_grid_shape
from robopoint.model.llava_arch import unpad_image
disable_torch_init()

from robopoint.minimal.utils import text2pixels, plot_points_on_image, prepare_multimodal_inputs_simplified


@torch.inference_mode()
def get_image_features(image_path: str, image_processor, model):
    if type(image_path) == str:
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path.convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    images = image_tensor.unsqueeze(0).half().cuda()                    # 1, 3, 336, 336
    image_features = model.get_model().get_vision_tower()(images)       # 1, 576 (24*24 patches), 1024
    image_features = model.get_model().mm_projector(image_features)     # 1, 576, 5120
    return image_features


@torch.inference_mode()
def process_one(image_path: str, qs: str, tokenizer, image_processor, model, conv_mode, temperature, num_beams, image_features=None):
    if DEFAULT_IMAGE_TOKEN not in qs:
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()   # 1, 142

    if image_features is None:
        image_features = get_image_features(image_path, image_processor, model)  # 1, 576, 5120

    inputs_embeds = prepare_multimodal_inputs_simplified(model, input_ids, image_features)

    output = super(LlamaForCausalLM, model).generate(
        position_ids=None,
        attention_mask=None,
        inputs_embeds=inputs_embeds,     # 1, 717, 5120
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        num_beams=num_beams,
        max_new_tokens=2048,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    output_ids = output.sequences[0]
    scores = output.scores
    min_len = min(len(output.sequences[0]), len(output.scores))
    confidences = []
    for token_id, score_tensor in zip(output_ids[:min_len], scores[:min_len]):
        probs = torch.softmax(score_tensor[0], dim=-1)
        confidence = probs[token_id].item()
        confidences.append(confidence)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return output_text, confidences


if __name__ == "__main__":
    model_path = "wentao-yuan/robopoint-v1-vicuna-v1.5-13b"
    image_path = "/robodata/smodak/repos/RoboPoint/chairs.png"
    question = "Identify locations on the floor in the vacant space between the two chairs. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image. Return empty list if no such points exist."

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )

    out_text, confs = process_one(
        image_path=image_path,
        qs=question,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
        conv_mode="llava_v1",
        temperature=0.0,
        num_beams=4
    )
    print(out_text)  # out is str (a list of tuples)
    # print(np.mean(confs))
    plot_points_on_image(image_path=image_path, answer=out_text, save_path='robopoint/minimal/output.png')
