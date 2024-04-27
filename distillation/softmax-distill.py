# Code adapted from https://huggingface.co/docs/transformers/main/en/tasks/knowledge_distillation_for_image_classification

from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from trl import SFTTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import login

class SoftMaxDistillationTrainer(SFTTrainer):
    def __init__(self, teacher_model=None, student_model=None, temperature=None, lambda_param=None,  *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher.to(device)
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
	return f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

### Task:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}
"""

def startDistillation():
    dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
    # Load the dataset
    ds = load_dataset(dataset_name, split="train")

    tiny_llama_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    code_llama_model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")

    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=tiny_llama_model)


    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    login(token="")



    training_args = TrainingArguments(
      output_dir="model-out",
      num_train_epochs=30,
      learning_rate=2e-5,
      per_device_train_batch_size=1,
      per_device_eval_batch_size=1,
      weight_decay=0.01,
      fp16=True,
      logging_dir="distilled-model/logs",
      logging_strategy="epoch",
      evaluation_strategy="epoch",
      save_strategy="epoch",
      load_best_model_at_end=True,
      push_to_hub=True,
      hub_strategy="every_save",
      hub_model_id="distill-code-tinyllama",
    )

    trainer = SoftMaxDistillationTrainer(
      teacher_model=code_llama_model,
      student_model=tiny_llama_model,
      args=training_args,
      train_dataset=ds,
      data_collator=data_collator,
      tokenizer=tokenizer,
      temperature=5,
      lambda_param=0.5,
      formatting_func=format_instruction,
      packing=True,
    )

    trainer.train()

startDistillation()
