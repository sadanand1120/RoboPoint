import torch
from PIL import Image
import math
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
import re

from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robopoint.conversation import conv_templates
from robopoint.model.builder import load_pretrained_model
from robopoint.utils import disable_torch_init
from robopoint.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
disable_torch_init()


def text2pixels(text, width=640, height=480):
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []
    for match in matches:
        vector = [
            float(num) if '.' in num else int(num) for num in match.split(',')
        ]
        if len(vector) == 2:
            x, y = vector
            if isinstance(x, float) or isinstance(y, float):
                x = int(x * width)
                y = int(y * height)
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0, x1, y1 = vector
            if isinstance(x0, float):
                x0 = int(x0 * width)
                y0 = int(y0 * height)
                x1 = int(x1 * width)
                y1 = int(y1 * height)
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = 1
            y, x = np.where(mask)
            points.extend(list(np.stack([x, y], axis=1)))
    return np.array(points)


def plot_points_on_image(image_path, answer, save_path='output.png'):
    img = Image.open(image_path)
    width, height = img.size
    points = text2pixels(answer, width, height)

    fig, ax = plt.subplots()
    ax.imshow(img)

    # Use default colormap
    colors = plt.cm.get_cmap('tab10')

    for i, (x, y) in enumerate(points):
        c1 = patches.Circle((x, y), 3, color=colors(3), fill=True)
        ax.add_patch(c1)
        c2 = patches.Circle((x, y), 10, color=colors(3), fill=False)
        ax.add_patch(c2)

    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def get_chunk(lst, n, k):
    """Split list into n roughly equal chunks and return the k-th chunk"""
    chunk_size = math.ceil(len(lst) / n)
    return lst[k * chunk_size: (k + 1) * chunk_size]


@torch.inference_mode()
def process_one(image_path: str, qs: str, tokenizer, image_processor, model, conv_mode, temperature, num_beams):
    if DEFAULT_IMAGE_TOKEN not in qs:
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    output = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).half().cuda(),
        image_sizes=[image.size],
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        num_beams=num_beams,
        max_new_tokens=1024,
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
    image_path = "chairs.png"
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
    print(np.mean(confs))
    plot_points_on_image(image_path=image_path, answer=out_text, save_path='minimal/output.png')
