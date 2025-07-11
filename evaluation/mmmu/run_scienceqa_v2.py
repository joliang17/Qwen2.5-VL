import os
CACHE_DIR = '/fs/nexus-faculty/zhou/colorbench/cache'

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

import sys
import json
import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Any
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor
import string
import traceback
import json
from openai import OpenAI

# Local imports from refactored files
from dataset_utils import load_dataset, load_dataset_json, dump_image, MMMU_preproc
from eval_utils import build_judge, eval_single_sample

from qwen2_vl.model import Qwen2VLChat
from qwen_vl_utils import process_vision_info

def run_inference(args):
    """Run inference on the MMMU dataset."""
    # Load dataset

    data = load_dataset_json(args.dataset_path)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Set up dump_image function
    def dump_image_func(line):
        return dump_image(line, img_root)
    
    # Set up CoT prompt if enabled
    cot_prompt = ""
    if args.use_cot:
        cot_prompt = args.cot_prompt if args.cot_prompt else " If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
        print(f"Using CoT prompt: {cot_prompt}")

    # Initialize HuggingFace model
    print(f"Loading HuggingFace model from {args.model_path}")
    model = Qwen2VLChat(
        model_path=args.model_path,
        temperature=0.01,
        top_p=0.001,
        top_k=1,
        use_custom_prompt=True,
        min_pixels=1280*28*28,
        max_pixels=5120*28*28
    )
    model.set_dump_image(dump_image_func)

    # Run inference
    results = []
    for i, sample in tqdm(enumerate(data), total=len(data), desc="Running inference"):
        index = sample['index']
        
        # Generate response using HuggingFace
        messages = model.build_prompt(sample, args.dataset)
        
        # Add CoT prompt if enabled
        if args.use_cot and len(messages) > 0 and messages[-1]['type'] == 'text':
            messages[-1]['value'] += cot_prompt
            
        response = model.generate(messages)
            
        print(f"response: {response}")
        print(f"annotation answer: {sample['conversations'][1]}")
        print('-' * 50)
        
        # Save result
        result = {
            "question_id": int(index) if isinstance(index, np.integer) else index,
            # "annotation": sample,
            "task": args.dataset,
            "result": {"gen": response},
            "messages": messages,
            "original_answer": sample['conversations'][1],
        }
        results.append(result)
        
        # Write intermediate results
        if i % 10 == 0:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=4)
            
    # Write final results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Inference completed. Results saved to {args.output_file}")


def api_judge(api_key, choices, answer, prediction, max_retries=3):
    client = OpenAI(api_key=api_key)

    # Original direct prompt
    prompt = "You will be given a list of choices, ground truth answer and a model prediction result. Check whether the model prediction result is true or not. Return Yes or No. " + f"Choices: {choices}\nAnswer: {answer}\nModel prediction: {prediction}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(2)  # Wait before retrying

    return "Error: Failed after multiple attempts"


def run_evaluation(args):
    """Run evaluation on inference results."""
    # Load results
    results = []
    with open(args.input_file, 'r') as f:
        list_results = json.load(f)

    for job in list_results:
        q_id = job['question_id']
        img_path = job['messages'][0]['value']
        question = job['messages'][1]['value']
        qs = question.split('\nChoices: ')[0].replace('Question: ', '')
        choices = question.split('\nChoices: ')[1]
        answer = job['original_answer']['value']
        prediction = job["result"]["gen"]
        results.append({"index": q_id, 'image': img_path, 'question': qs, 'choices': choices, 'answer': answer, 'prediction': prediction})
            
    list_correct = []
    list_incorrect = []
    for data in tqdm(results):
        if data['prediction'] == data['answer']:
            list_correct.append(data)
        else:
            api_response = api_judge(args.api_key, data['choices'], data['answer'], data['prediction'], max_retries=3)

            if api_response.lower() == 'yes':
                list_correct.append(data)
            else:
                list_incorrect.append(data)

    # Calculate overall accuracy
    accuracy = len(list_correct) / len(results)
    print(f"Evaluation completed. Overall accuracy: {accuracy:.4f}")

    # Write final results
    os.makedirs(args.output_folder, exist_ok=True)
    with open(f"{args.output_folder}/{args.result_type}_correct.json", 'w') as f:
        json.dump(list_correct, f, indent=4)
    with open(f"{args.output_folder}/{args.result_type}_incorrect.json", 'w') as f:
        json.dump(list_incorrect, f, indent=4)
    
    print(f"Results saved to {args.output_folder}")

def main():
    parser = argparse.ArgumentParser(description="MMMU Evaluation Script")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    infer_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    infer_parser.add_argument("--dataset-path", type=str, help="The absolute path of MMMU_DEV_VAL.tsv")
    infer_parser.add_argument("--data-dir", type=str, help="The absolute path of MMMU_DEV_VAL.tsv")
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    infer_parser.add_argument("--use-cot", action="store_true", help="Use Chain-of-Thought prompting")
    infer_parser.add_argument("--cot-prompt", type=str, default="", help="Custom Chain-of-Thought prompt")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--input-file", type=str, required=False, help="Input file with inference results")
    eval_parser.add_argument("--output-folder", type=str, required=False, help="Output file path", default="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/mcq_parsed")
    eval_parser.add_argument("--api-key", type=str, required=False, help="Input file with inference results")
    eval_parser.add_argument("--result-type", type=str, default='mcg', required=False, help="Input file with inference results")
    
    args = parser.parse_args()
    
    if args.mode == "infer":
        os.environ['LMUData'] = args.data_dir
        run_inference(args)
    elif args.mode == "eval":
        run_evaluation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
