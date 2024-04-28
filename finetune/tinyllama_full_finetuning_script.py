# Install Requirements First
from datasets import load_dataset
from random import randrange

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from trl import SFTTrainer

# The model that you want to train from the Hugging Face hub
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# The instruction dataset to use
dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
#dataset_name = "HuggingFaceH4/CodeAlpaca_20K"
# Dataset split
new_model = "full_finetuned-tinyllama"
dataset_split= "train"
# Load the entire model on the GPU 0
device_map = {"": 0}

################################################################################
# TrainingArguments parameters
################################################################################
# Output directory where the model predictions and checkpoints will be stored
output_dir = new_model
# Number of training epochs
num_train_epochs = 1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False 
bf16 = True
# Batch size per GPU for training
per_device_train_batch_size = 4
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1 # 2
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4 #1e-5
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule
lr_scheduler_type = "cosine" #"constant"
# Number of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = False
# Save checkpoint every X updates steps
save_steps = 0
# Log every X updates steps
logging_steps = 25
# Disable tqdm
disable_tqdm= False

################################################################################
# SFTTrainer parameters
################################################################################
# Maximum sequence length to use
max_seq_length = 2048 #None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = True #False

# Load dataset from the hub
dataset = load_dataset(dataset_name, split=dataset_split)
# Show dataset size
print(f"dataset size: {len(dataset)}")
# Show an example
print(dataset[randrange(len(dataset))])

def format_instruction(sample):
	return f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

### Task:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}
"""

# Load the pretrained model
model = AutoModelForCausalLM.from_pretrained(model_id, use_cache = False, device_map=device_map, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.config.pretraining_tp = 1

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Define the training arguments
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size, # 6 if use_flash_attention else 4,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    optim=optim,
    #save_steps=save_steps,
    logging_steps=logging_steps,
    save_strategy="epoch",
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    #max_steps=max_steps,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    disable_tqdm=disable_tqdm,
    report_to="tensorboard",
    seed=42
)

# Create the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=packing,
    formatting_func=format_instruction,
    args=args,
)

# train
trainer.train() # there will not be a progress bar since tqdm is disabled

# save model in local
trainer.save_model("finetuned-tinyllama")

