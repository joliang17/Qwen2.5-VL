import sys
import os
from argparse import ArgumentParser

from tqdm import tqdm, trange
from PIL import Image
from typing import Dict
import json
import random
from sklearn.model_selection import train_test_split

random.seed(42)

def _clean_q(s: str) -> str:
    # remove lone <image> tokens and surrounding whitespace/newlines
    return s.replace("\n<image>", "").replace("<image>", "").strip()


def prepare_samples(s_type: str='keywords', img_path: str='', ori_question: str='', gt_answer: str='', keywords: str=''):
    dict_sample = {"image": img_path}
    clean_q = _clean_q(ori_question)

    dict_sample["conversations"] = []
    if s_type == 'keywords':
        question = f"<image>\nGiven the image and keywords, generate the correct answers. Keywords: {keywords}"
        dict_sample["conversations"].append({"from": "human", "value": question})
        answer = f"{clean_q}\nAnswer: {gt_answer}"
        dict_sample["conversations"].append({"from": "gpt", "value": answer})
    else:
        question = f"<image>\nGiven the image and question, generate the correct answers. Question: {clean_q}"
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


def save_random_subsets(list_keywords, list_normal, saved_folder: str, sizes=(2000, 4000)):
    """
    Randomly select subsets (e.g., 2k, 4k) from list_keywords and their corresponding list_normal.
    Save them as new json files in the saved_folder.
    """
    os.makedirs(saved_folder, exist_ok=True)

    total = len(list_keywords)
    assert len(list_keywords) == len(list_normal), "list_keywords and list_normal must have same length"

    for size in sizes:
        if size > total:
            print(f"Warning: requested size {size} is larger than dataset {total}, skipping.")
            continue

        # sample indices
        sampled_indices = random.sample(range(total), size)

        sampled_keywords = [list_keywords[i] for i in sampled_indices]
        sampled_normal = [list_normal[i] for i in sampled_indices]

        # save files
        with open(os.path.join(saved_folder, f"train_keywords_{size}.json"), "w") as f:
            json.dump(sampled_keywords, f, indent=4)

        with open(os.path.join(saved_folder, f"train_normal_{size}.json"), "w") as f:
            json.dump(sampled_normal, f, indent=4)

        print(f"Saved {size} samples to train_keywords_{size}.json and train_normal_{size}.json")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--json_name", type=str, default="qwen3_32b")
    parser.add_argument("--saved_folder", type=str, default="../evaluation/mmmu/scienceqa_keywords")
    parser.add_argument("--root_folder", type=str, default="/fs/nexus-scratch/yliang17/Research/VLM/task_generation/keyword_results_scienceqa_test")
    parser.add_argument("--image_folder", type=str, default="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/scienceqa/images")
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

    save_random_subsets(list_keywords, list_normal, saved_folder, sizes=(1000, 2000, 3000, 4000, 5000))