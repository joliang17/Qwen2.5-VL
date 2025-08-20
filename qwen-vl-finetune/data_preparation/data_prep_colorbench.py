import sys
import os

from tqdm import tqdm, trange
from PIL import Image
from typing import Dict, List
import json
from sklearn.model_selection import train_test_split


def prepare_samples(s_type: str='keywords', img_path: str='', ori_question: str='', gt_answer: str='', keywords: str=''):
    dict_sample = {"image": img_path}
    dict_sample["conversations"] = []
    if s_type == 'keywords':
        question = f"<image>\nGiven the image and keywords, generate the correct answers. Keywords: {keywords}"
        dict_sample["conversations"].append({"from": "human", "value": question})
        answer = f"Question: {ori_question}\nAnswer: {gt_answer}"
        dict_sample["conversations"].append({"from": "gpt", "value": answer})
    else:
        question = f"<image>\n{ori_question}"
        dict_sample["conversations"].append({"from": "human", "value": question})
        dict_sample["conversations"].append({"from": "gpt", "value": gt_answer})
    return dict_sample


def prepare_dataset(list_samples: List):
    list_formated = []
    for img_id, dict_meta in enumerate(list_samples):
        img_path = dict_meta['new_image']
        ori_question = dict_meta['prompt']
        gt_answer = dict_meta['answer']

        dict_sample = prepare_samples(s_type='normal', img_path=img_path, ori_question=ori_question, gt_answer=gt_answer, )
        list_formated.append(dict_sample)

    return list_formated

if __name__ == "__main__":
    json_folder = '/fs/nexus-scratch/yliang17/Research/VLM/ColorBench/Eval_json_collected_0724/'
    list_json = os.listdir(json_folder)
    list_all = []
    for json_file in list_json:
        with open(os.path.join(json_folder, json_file), "r") as f:
            list_samples = json.load(f)
        list_all.extend(list_samples)

    list_formated = prepare_dataset(list_samples=list_all)

    saved_folder = 'colorbench'
    os.makedirs(saved_folder, exist_ok=True)

    with open(f"{saved_folder}/train.json", "w") as f:
        json.dump(list_formated, f, indent=4)


