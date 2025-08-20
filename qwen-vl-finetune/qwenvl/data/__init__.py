import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

SCIENCE_QA = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa/mcq.json",
    "data_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa/images",
}

SCIENCE_QA_NORMAL = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa/normal.json",
    "data_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa/images",
}

SCIENCE_QA_KEYWORDS_1K = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_keywords_1000.json",
    "data_path": "",
}

SCIENCE_QA_NORMAL_V2_1K = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_normal_1000.json",
    "data_path": "",
}

SCIENCE_QA_KEYWORDS_2K = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_keywords_2000.json",
    "data_path": "",
}

SCIENCE_QA_NORMAL_V2_2K = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_normal_2000.json",
    "data_path": "",
}

SCIENCE_QA_KEYWORDS_3K = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_keywords_4000.json",
    "data_path": "",
}

SCIENCE_QA_NORMAL_V2_3K = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_normal_4000.json",
    "data_path": "",
}

SCIENCE_QA_KEYWORDS_4K = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_keywords_4000.json",
    "data_path": "",
}

SCIENCE_QA_NORMAL_V2_4K = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_normal_4000.json",
    "data_path": "",
}

SCIENCE_QA_KEYWORDS_5K = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_keywords_4000.json",
    "data_path": "",
}

SCIENCE_QA_NORMAL_V2_5K = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_normal_4000.json",
    "data_path": "",
}

SCIENCE_QA_KEYWORDS = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_keywords.json",
    "data_path": "",
}

SCIENCE_QA_NORMAL_V2 = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/scienceqa_keywords/train_normal.json",
    "data_path": "",
}

MMINSTRUCT_KEYWORDS = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/mminstruct/train_keywords.json",
    "data_path": "",
}

MMINSTRUCT = {
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/mminstruct/train_normal.json",
    "data_path": "",
}

COLORBENCH = {  
    "annotation_path": "/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/qwen-vl-finetune/colorbench/train.json",
    "data_path": "",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "scienceqa": SCIENCE_QA,
    "scienceqa_normal": SCIENCE_QA_NORMAL,
    "scienceqa_keywords": SCIENCE_QA_KEYWORDS,
    "scienceqa_normal_v2": SCIENCE_QA_NORMAL_V2,

    # ScienceQA keyword/normal subsets
    "scienceqa_keywords_1k": SCIENCE_QA_KEYWORDS_1K,
    "scienceqa_normal_v2_1k": SCIENCE_QA_NORMAL_V2_1K,
    "scienceqa_keywords_2k": SCIENCE_QA_KEYWORDS_2K,
    "scienceqa_normal_v2_2k": SCIENCE_QA_NORMAL_V2_2K,
    "scienceqa_keywords_3k": SCIENCE_QA_KEYWORDS_2K,
    "scienceqa_normal_v2_3k": SCIENCE_QA_NORMAL_V2_2K,
    "scienceqa_keywords_4k": SCIENCE_QA_KEYWORDS_4K,
    "scienceqa_normal_v2_4k": SCIENCE_QA_NORMAL_V2_4K,
    "scienceqa_keywords_5k": SCIENCE_QA_KEYWORDS_4K,
    "scienceqa_normal_v2_5k": SCIENCE_QA_NORMAL_V2_4K,

    "mminstruct_keywords": MMINSTRUCT_KEYWORDS,
    "mminstruct": MMINSTRUCT,
    "colorbench": COLORBENCH, 
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    # dataset_names = ["scienceqa_normal", "scienceqa"]
    dataset_names = [
        "scienceqa_keywords_1k",
        "scienceqa_normal_v2_1k",
        "scienceqa_keywords_2k",
        "scienceqa_normal_v2_2k",
        "scienceqa_keywords_3k",
        "scienceqa_normal_v2_3k",
        "scienceqa_keywords_4k",
        "scienceqa_normal_v2_4k",
        "scienceqa_keywords_5k",
        "scienceqa_normal_v2_5k",
        "mminstruct_keywords",
        "mminstruct",
        "colorbench"
    ]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
