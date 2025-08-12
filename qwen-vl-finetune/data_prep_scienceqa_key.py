import sys
import os
from argparse import ArgumentParser

from tqdm import tqdm, trange
from PIL import Image
from typing import Dict
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
        question = f"<image>\nGiven the image and question, generate the correct answers. Question: {ori_question}"
        dict_sample["conversations"].append({"from": "human", "value": question})
        answer = f"Answer: {gt_answer}"
        dict_sample["conversations"].append({"from": "gpt", "value": answer})
    return dict_sample


def prepare_dataset(dict_samples: dict, image_folder: str):
    list_normal = []
    list_keywords = []
    for img_id, list_meta in tqdm(dict_samples.items()):
        img_path = list_meta[-1]
        qa_pairs = list_meta[1:-1]

        # # process image path
        # img_path = os.path.join(img_folder, img_name)
        if not os.path.exists(img_path):
            print(f"Image {img_path} does not exist, skip")
            continue

        for qa in qa_pairs:
            ori_question = qa['question'].replace('<image>\n', ' ')
            gt_answer = qa['answer']
            keywords = qa['content']

            dict_sample = prepare_samples(s_type='normal', img_path=img_path, ori_question=ori_question, gt_answer=gt_answer, keywords=keywords)
            list_normal.append(dict_sample)
            dict_sample = prepare_samples(s_type='keywords', img_path=img_path, ori_question=ori_question, gt_answer=gt_answer, keywords=keywords)
            list_keywords.append(dict_sample)

    return list_normal, list_keywords

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--json_name", type=str, default="qwen3_32b")
    parser.add_argument("--saved_folder", type=str, default="scienceqa_keywords")
    parser.add_argument("--root_folder", type=str, default="/fs/nexus-scratch/yliang17/Research/VLM/task_generation")
    parser.add_argument("--image_folder", type=str, default="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa/images")
    args = parser.parse_args()

    image_folder = args.image_folder
    json_file = os.path.join(args.root_folder, f"{args.json_name}.json")
    saved_folder = args.saved_folder

    # {"id": [caption, qa_pairs, image_path]}
    with open(json_file, "r") as f:
        dict_keywords = json.load(f)

    list_normal, list_keywords = prepare_dataset(dict_samples=dict_keywords, image_folder=image_folder, )

    os.makedirs(saved_folder, exist_ok=True)
    with open(f"{saved_folder}/train_keywords.json", "w") as f:
        json.dump(list_keywords, f, indent=4)

    with open(f"{saved_folder}/train_normal.json", "w") as f:
        json.dump(list_normal, f, indent=4)
