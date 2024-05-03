 = Experiments
// Full description of your experimental design, results, and analyses. Whereas a typical experiments section focuses on key metrics for the benchmark task on the full dataset, yours should reflect the research diary nature of this document. For example, if you ran a pilot experiment on a fraction of the dataset (e.g. 1/10), you should include those results here. You should also include results with timing information about your experiments, e.g. an estimate of how long it takes to train the model for one epoch and to convergence, an estimate of how long it takes to validate the model on the val/test sets. Also, describe what hardware you are using (e.g. specs of your CPU or GPU).

== Replication of Baseline Model
We successfully replicated the reported results of the DeepSeek-Coder 1.5B model on the HumanEval and MBPP datasets. Our results are close to the reported results, but consistently about 5% lower likely due to stochasticity in sampling and differences in the implementation of evaluation of generated Python programs. We additionally, consider TinyLlama-Chat 1.1B and CodeLlama-Python 7B as our baselines for finetuning and distillation, and evaluated those models on HumanEval and MBPP as well. All baseline evaluation results are present in @baselines.
#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: horizon,
    table.header(
      [], [HumanEval pass@1], [MBPP pass@1],
    ),
    "DeepSeek 1.3B",
    "59.76%",
    "44.80%",
    "TinyLlama-Chat 1.1B",
    "11.59%",
    "0.60%",
    "CodeLlama-Python 7B",
    "38.40%",
    "47.60%",
    "Quantized CodeLlama",
    "39.63%",
    "43.60%"
  ),
  caption: "Replicated results for baseline models on HumanEval and MBPP"
) <baselines>



== Supervised Finetuning of TinyLlama
Our  first approach was to perform supervised finetuning. We started with finetuning TinyLlama-1.1B with Quantized-LoRA for one epoch on the `iamtarun/python_code_instructions_18k_alpaca` dataset. As we didn't observe an improvement when evaluating on HumanEval, we decided to perform full-finetuning. After one epoch, the training loss had not seemed to converge, so we trained for 4 epochs. Results for the experiments are in @finetuned
#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: horizon,
    table.header(
      [], [HumanEval pass@1], [MBPP pass@1] 
    ),
    "TinyLlama-Chat 1.1B (Base)",
    "11.59%","0.60%",
    "Fine-Tuned w/ QLoRA",
    "11.59%", "6.80%",
    "Fully Fine-Tuned",
    "14.63%", "5.00%"
  ),
  caption: "Finetuned TinyLlama evaluation on HumanEval"
) <finetuned>

== Distillation of Code Llama
Using the distillation techniques outlined above, we have distilled Code Llama - 13B into our fine-tuned version of TinyLlama, representing a tenfold reduction in parameters. Using this new model, our evaluated performance on the HumanEval and MBPP datasets can be found in @distilled.

 #figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: horizon,
    table.header(
      [], [HumanEval pass@1], [HumanEval pass@2],
    ),
    "DeepSeek-1.3B",
    "65.2%",
    "--",
    "CodeLlama-Python-7B",
    "38.4%",
    "--",
    "TinyLlama (Base)",
    "11.6%",
    "--",
    "TinyLlama (Fully Fine-Tuned)",
    "14.6%",
    "15.2%",
    "Distilled CodeLlama to TinyLlama(Best Results)",
    "11.59%",
    "--"
  ),
  caption: "A comparison of small models on HumanEval"
) <distilled>

== Enhancing our Fine-tuned Model

Using the techniques outlined in the Codet paper, we further enhance the capability of our fine-tuned model. Our results can be found in @enhanced.

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: horizon,
    table.header(
      [], [HumanEval pass@1], [HumanEval pass@2],
    ),
    "Fine-Tuned",
    "14.6%",
    "15.2%",
    "Fine-Tuned + CodeT",
    "15.8%",
    "20.4%"
  ),
  caption: "A comparison of small models on HumanEval"
) <enhanced>
