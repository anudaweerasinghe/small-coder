# Code adapted from https://huggingface.co/docs/transformers/main/en/tasks/knowledge_distillation_for_image_classification

from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from trl import SFTTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import login
import wandb

import os

model_name = "distilled-finetuned-code-llama"

os.environ["WANDB_PROJECT"] = "tiny-code-llama"
os.environ["WANDB_LOG_MODEL"] = "false"  # don't log model checkpoints

class SoftMaxDistillationTrainer(SFTTrainer):
    def __init__(self, teacher_model=None, student_model=None, temperature=None, lambda_param=None,  *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.teacher.eval()
        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, student, inputs, return_outputs=False):
        student_output = self.student(**inputs)

        with torch.no_grad():
          teacher_output = self.teacher(**inputs)

        # Compute soft targets for teacher and student
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        # Compute the loss
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # Compute the true label loss
        student_target_loss = student_output.loss

        # Calculate final loss
        loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss


# Set the instruction format for iamtarun/python_code_instructions_18k_alpaca
def format_instruction(sample):

    outputs = []

    for i in range(len(sample['output'])):
        outputs.append(f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

### Task:
{sample['instruction'][i]}

### Input:
{sample['input'][i]}

### Response:
{sample['output'][i]}
""")

    return outputs

def startDistillation():
    dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
    # Load the dataset
    ds = load_dataset(dataset_name, split="train")

    finetuned_tiny_llama_model = AutoModelForCausalLM.from_pretrained("anudaw/full_finetuned-code-tinyllama", torch_dtype=torch.bfloat16)
    code_llama_model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf", load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained("anudaw/full_finetuned-code-tinyllama")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    login(token=os.environ.get("HF_TOKEN"))
    wandb.login(key=os.environ.get("WANDB_KEY"))


    training_args = TrainingArguments(
      output_dir="model-out",
      num_train_epochs=3,
      learning_rate=2e-5,
      per_device_train_batch_size=1,
      per_device_eval_batch_size=1,
      weight_decay=0.001,
      fp16=False,
      bf16=True,
      logging_dir="distilled-model/logs",
      logging_strategy="steps",
      logging_steps=100,
      save_strategy="epoch",
      # load_best_model_at_end=True,
      push_to_hub=True,
      hub_strategy="every_save",
      hub_model_id=model_name,
      report_to="wandb",
      lr_scheduler_type="constant",
      warmup_ratio=0.03,
      gradient_accumulation_steps=32,
      gradient_checkpointing=True,
    )

    trainer = SoftMaxDistillationTrainer(
      teacher_model=code_llama_model,
      student_model=finetuned_tiny_llama_model,
      args=training_args,
      train_dataset=ds,
      data_collator=data_collator,
      tokenizer=tokenizer,
      temperature=3,
      lambda_param=0.1,
      formatting_func=format_instruction,
      packing=False,
    )

    trainer.train()

startDistillation()