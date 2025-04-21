from PIL import Image
from tqdm import tqdm
import json
import numpy as np
import re
from robopoint.minimal.utils import text2pixels


if __name__ == '__main__':
    question_file = "datasets/where2place/point_questions.jsonl"
    answer_file = "output/robopoint-v1-vicuna-v1.5-13b.jsonl"
    data_dir = "datasets/where2place"

    with open(question_file, "r") as f:
        questions = [json.loads(line) for line in f.readlines()]

    with open(answer_file, "r") as f:
        answers = [json.loads(line) for line in f.readlines()]

    accuracy = []
    for idx, question in enumerate(tqdm(questions)):
        try:
            points = text2pixels(answers[idx]['text'])
        except:
            points = np.array([])
        mask = np.array(Image.open(f"{data_dir}/masks/{idx:02d}.jpg")) / 255.
        acc = 0
        if len(points) > 0:
            in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) \
                & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
            acc = np.concatenate([
                mask[points[in_range, 1], points[in_range, 0]],
                np.zeros(points.shape[0] - in_range.sum())
            ]).mean()
        accuracy.append(acc)
    print(f"Accuracy: {np.mean(accuracy)}")
