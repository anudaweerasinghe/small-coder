= Dataset and Task
// Describe the dataset you will use, a precise definition of the task, and what metric(s) you will use for evaluation
== Dataset
HumanEval and MBPP, which are both collections of natural language prompts, code that solves the prompt, and test cases to verify the correctness of generated code. 
== Task 
Given a natural language description of a problem, output an implementation of a solution to the problem in Python.
== Evaluation Metrics
pass@1 and pass@10, which measures the percentage of problems for which the model generates a correct solution, i.e. all test cases pass, within the first attempt, and first 10 attempts respectively. 