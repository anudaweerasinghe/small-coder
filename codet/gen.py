from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("anudaw/full_finetuned-code-tinyllama", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("anudaw/full_finetuned-code-tinyllama", trust_remote_code=True).cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 1024

from tqdm import tqdm
import numpy as np
import math

def generate_results(model, tokenizer, entries, output_file, num_samples):
  results = []

  num_entries = len(entries)
  batch_size = 20

  for (i, row) in tqdm(entries.iterrows(), total=len(entries)):
    prompt = row['prompt']
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    cur_list = []
    
    for j in range(num_samples // batch_size):
      with torch.no_grad():
        output_ids = model.generate(**input_ids, max_length=max_length, do_sample=True, temperature=0.5, num_return_sequences=batch_size)
    
        for output in output_ids:
          generated_text = tokenizer.decode(output, skip_special_tokens=True)
          cur_list.append(generated_text)

    json.dump({ 'prompt': prompt, 'samples': cur_list }, output_file)
    output_file.write('\n')
    cur_list = []

import pandas as pd
from pathlib import Path
import json

dataset_folder = Path('../../CodeT/CodeT/data/dataset')
dataset_name = 'HumanEval'

codegen_file = dataset_folder / f'{dataset_name}_for_code_generation.jsonl'
testcase_file = dataset_folder / f'{dataset_name}_for_test_case_generation.jsonl'

codegen_json = pd.read_json(path_or_buf=Path(codegen_file), lines=True)
testcase_json = pd.read_json(path_or_buf=Path(testcase_file), lines=True)

with open(f'{dataset_name}-codegen-20-temp0.5-n60-p2.jsonl', mode='w') as writer:
    results = generate_results(model, tokenizer, codegen_json.iloc[89:], writer, num_samples=60)
    
# with open(f'{dataset_name}-testcase-20-temp0.5-n60.jsonl', mode='w') as writer:
    # results = generate_results(model, tokenizer, testcase_json, writer, num_samples=60)