= Dataset and Task
// Detailed description of the task, dataset, and metric(s) for evaluation. Description of the model.

== Dataset
We will evaluate our methods on HumanEval by OpenAI, and MBPP, which is crowd-sourced. Both of these datasets are collections of natural language prompts, code that solves the prompt, and test cases to verify the correctness of generated code. 

== Task 
Given a natural language description of a problem, output an implementation of a solution to the problem in Python.

== Evaluation Metrics
pass@$k$,which measures the percentage of problems for which the model generates a correct solution, i.e. all test cases pass, within the first $k$ attempts. We evaluate this for $k=1$ and $2$ for HumanEval and MBPP.

== Model
The state of the art model we are using as a baseline is `DeepSeek-Coder-1.3B`, a model with 1.3B parameters trained on open source code on Github trained with next token prediction and fill-in-the-middle training strategies.

The base model we are using for our experiments is TinyLlama, a parameter with 1.1B parameters structured similarly to the larger Llama 2 models that shares the same tokenizer.