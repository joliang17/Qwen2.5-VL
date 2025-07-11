import sys
import os

CACHE_DIR = "/fs/nexus-projects/wilddiffusion/cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

from tqdm import tqdm, trange
from datasets import load_dataset
from PIL import Image
import json

# ds = load_dataset("derek-thomas/ScienceQA", split='train')
ds = load_dataset("derek-thomas/ScienceQA", split='test')

saved_folder = '../evaluation/mmmu/scienceqa'
os.makedirs(saved_folder, exist_ok=True)
os.makedirs(f"{saved_folder}/images", exist_ok=True)

list_mcq = []
list_normal = []
for d_id, data in enumerate(tqdm(ds)):
    # convert to mcq generation task
    img_file = data['image']
    if img_file is None:
        continue
    img_path = f"{saved_folder}/images/{d_id}.png"
    img_file.save(img_path)

    # preprocessing
    ori_question = data['question']
    list_choices = data['choices']
    num_choices = len(list_choices)
    gt_answer = list_choices[data['answer']]
    choices_str = '\t'.join(str(c) for c in list_choices)  # ✅ correct

    # add conversation for mcq generation
    dict_mcq = {"image": img_path}
    dict_mcq["conversations"] = []
    question = f"Generate the question, {num_choices} choices and a correct answer based on the given image\n<image>"
    dict_mcq["conversations"].append({"from": "human", "value": question})
    answer = f"Question: {ori_question}\nChoices: {choices_str}\nAnswer: {gt_answer}"
    dict_mcq["conversations"].append({"from": "gpt", "value": answer})
    list_mcq.append(dict_mcq)

    # add conversation for answer generation
    dict_normal = {"image": img_path}
    dict_normal["conversations"] = []
    gt_answer = data['choices'][data['answer']]
    dict_normal["conversations"].append({"from": "human", "value": f"Question: {ori_question}\nChoices: {choices_str}\n<image>"})
    dict_normal["conversations"].append({"from": "gpt", "value": gt_answer})
    list_normal.append(dict_normal)


mcq_file = f"{saved_folder}/mcq.json"
print(f"write to file {mcq_file}")
with open(mcq_file, "w") as f:
    json.dump(list_mcq, f, indent=4)


normal_file = f"{saved_folder}/normal.json"
print(f"write to file {normal_file}")
with open(normal_file, "w") as f:
    json.dump(list_normal, f, indent=4)

