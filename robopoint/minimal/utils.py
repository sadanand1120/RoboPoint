import torch
from PIL import Image
import math
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
import re

from robopoint.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from robopoint.utils import disable_torch_init
disable_torch_init()


def prepare_multimodal_inputs_simplified(model, input_ids, image_features):
    """
    Build a batch of `inputs_embeds` where every token equal to
    IMAGE_TOKEN_INDEX is replaced by its corresponding `image_features`.

    Args
    ----
    model           : the wrapped Llama/transformer model (must expose
                      `model.get_model().embed_tokens` and `model.config`)
    input_ids       : LongTensor of shape (B, L)
    image_features  : sequence of FloatTensors produced by the vision
                      tower; consumed in the order the image-tokens appear

    Returns
    -------
    inputs_embeds   : FloatTensor of shape (B, L_max, D) padded on the side
                      specified by `model.config.tokenizer_padding_side`
    """
    device = input_ids.device
    embed_tokens = model.get_model().embed_tokens
    pad_side = getattr(model.config, "tokenizer_padding_side", "right")
    max_len_cfg = getattr(model.config, "tokenizer_model_max_length", None)

    batch_embeds = []
    img_ptr = 0

    for ids in input_ids:                                 # iterate over batch
        img_pos = torch.where(ids == IMAGE_TOKEN_INDEX)[0].tolist()

        if not img_pos:                                   # no image tokens
            batch_embeds.append(embed_tokens(ids))
            continue

        # boundaries around every image token: [-1, i0, i1, ..., iN, L]
        bounds = [-1] + img_pos + [ids.shape[0]]
        segs = []

        for i in range(len(bounds) - 1):
            txt_ids = ids[bounds[i] + 1: bounds[i + 1]]  # text chunk
            if txt_ids.numel():
                segs.append(embed_tokens(txt_ids))
            if i < len(img_pos):                          # insert image feats
                segs.append(image_features[img_ptr])
                img_ptr += 1

        batch_embeds.append(torch.cat(segs, dim=0))

    # optional truncation to model’s hard limit
    if max_len_cfg is not None:
        batch_embeds = [e[: max_len_cfg] for e in batch_embeds]

    # pad to common length
    max_len = max(e.shape[0] for e in batch_embeds)
    embed_dim = batch_embeds[0].shape[1]
    padded = []

    for e in batch_embeds:
        pad = max_len - e.shape[0]
        pad_tensor = torch.zeros(pad, embed_dim, dtype=e.dtype, device=device)
        padded.append(torch.cat((pad_tensor, e), dim=0) if pad_side == "left"
                      else torch.cat((e, pad_tensor), dim=0))

    inputs_embeds = torch.stack(padded, dim=0)            # (B, L_max, D)
    return inputs_embeds


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


def get_chunk(lst, n, k):
    """Split list into n roughly equal chunks and return the k-th chunk"""
    chunk_size = math.ceil(len(lst) / n)
    return lst[k * chunk_size: (k + 1) * chunk_size]


def plot_points_on_image(image_path, answer, save_path='output.png'):
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('RGB')
    else:
        img = image_path.convert('RGB')
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
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def prepare_inputs_labels_for_multimodal(model, input_ids, image_features, position_ids=None, attention_mask=None, past_key_values=None, labels=None):
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = model.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = model.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(model.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)
        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    tokenizer_model_max_length = getattr(model.config, 'tokenizer_model_max_length', None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(model.config, 'tokenizer_padding_side', 'right') == "left":
            new_input_embeds_padded.append(torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                cur_new_embed
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
