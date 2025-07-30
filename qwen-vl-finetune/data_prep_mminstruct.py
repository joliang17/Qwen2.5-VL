import sys
import os

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


def prepare_dataset(dict_samples: dict=Dict):
    list_normal = []
    list_keywords = []
    for img_id, list_meta in tqdm(dict_samples.items()):
        img_name = list_meta[-1]
        qa_pairs = list_meta[1:-1]

        # process image path
        img_path = os.path.join(mm_img, img_name)
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
    mm_img = '/fs/nexus-projects/wilddiffusion/task_generation/images'
    mm_instruct_keywords = '/fs/nexus-projects/wilddiffusion/task_generation/keyword_results/qwen3_32b.json'

    # {"id": [caption, qa_pairs, image_path]}
    with open(mm_instruct_keywords, "r") as f:
        dict_keywords = json.load(f)

    # Split keys into train and test
    keys = list(dict_keywords.keys())
    train_keys, test_keys = train_test_split(keys, test_size=0.1, random_state=42)

    # Subsets
    dict_keywords_train = {k: dict_keywords[k] for k in train_keys}
    dict_keywords_test = {k: dict_keywords[k] for k in test_keys}

    list_normal_train, list_keywords_train = prepare_dataset(dict_samples=dict_keywords_train)
    list_normal_test, list_keywords_test = prepare_dataset(dict_samples=dict_keywords_test)

    saved_folder = 'mminstruct'
    os.makedirs(saved_folder, exist_ok=True)

    with open(f"{saved_folder}/train_normal.json", "w") as f:
        json.dump(list_normal_train, f, indent=4)

    with open(f"{saved_folder}/train_keywords.json", "w") as f:
        json.dump(list_keywords_train, f, indent=4)

    saved_folder = '../evaluation/mmmu/mminstruct'
    os.makedirs(saved_folder, exist_ok=True)

    with open(f"{saved_folder}/test_normal.json", "w") as f:
        json.dump(list_normal_test, f, indent=4)

    with open(f"{saved_folder}/test_keywords.json", "w") as f:
        json.dump(list_keywords_test, f, indent=4)
