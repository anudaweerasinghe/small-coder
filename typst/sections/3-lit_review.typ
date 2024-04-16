= Related Work
// Provide a short literature survey of 4 or more relevant papers. These papers should be relevant to your approach; they do not need to be relevant to the particular task/dataset that you chose.
== Code Llama: Open Foundation Models for Code @roziere2023code
Code Llama is a family of LLMs for code generation that were based on the Llama 2 models. They have 7B, 13B, 34B, and 70B parameter variants of three models - Code Llama, CodeLlama Python, Code Llama Instruct. The models are trained on a proprietary dataset, and ~14,000 synthetically generated interview question, solution, and unit test examples. Code Llama - Python performs the best on Human Eval and MBPP, achieving ~60%, ~85%, and \~95% on pass@1, pass@10, and pass@100 respectively. Code-Llama Python, which was trained on a 100B extra tokens of Python-heavy data, outperformed larger base versions of the model. 

== Distilling the Knowledge in a Neural Network @hinton2015distilling
Discusses an approach to transferring/distilling knowledge from a large pre-trained model to a small model that is easier/cheaper to deploy. Since the large models has many more parameters than the small model, and we don't have a good understanding of the meaning of each parameter, we cannot directly map from the larger space of parameters to the smaller one. Instead, Hinton et al's approach uses the fact that the final softmax values of the larger model convey important information about how the model is generalizing the problem. They use this insight, to train the smaller model with an objective of matching the larger model's final softmax values. The loss function of the smaller model is a weighted average of the cross entropy loss with respect to two different versions of the larger model's softmax values - one with a softer, more uniform distribution that's adjusted using the temperature value used to compute the softmax. Since the new objective is only based on the output of the larger model, the smaller model can be trained on unlabeled data. The distilled version of the small model outperforms the baseline version that's trained from scratch across all tasks presented in the paper, and even comes close to matching the larger model's performance on MNIST. \
\
The smaller model can also be a _specialist_ version of the larger model that is only predicting from a subset of the larger model's classes. The distillation process is the same except that all softmax values corresponding to classes not in the smaller model's target classes are assigned to a single _dustbin_ class. 
== MiniLM: Deep self-attention distillation @wang2020minilm 
An adaptation of Hinton et al's work for transformers with self-attention. Instead of distilling the final softmax values, they propose distilling the self-attention module of the last transformer layer of the teacher. It retains $>99%$ performance with \~50% of the parameters.  
== Codet: Code generation with generated tests @chen2022codet
This paper improves the quality of code generation by sampling unit tests from the same LM, and using them to sample better code. For a given input, begin by sampling solutions ${x_1, x_2, dots x_n}$. Then use the same LM to sample unit tests for the same input problem ${y_1, y_2, dots y_m}$. Run each test $y_i$ on each piece of code $x_j$, and form consensus set $cal(S)_(x_j)$ with all the tests that $x_j$ passes. Also form consensus set $cal(S)_(y_i)$ with all pieces of code that passes test $y_i$. Then create $K$ consensus sets $cal(S) = {(x, y) | x in cal(S)_x, y in cal(S_y)}$, each of which has score $|cal(S)_x||cal(S)_y|$. Run this process multiple times, and return a random sample $x$ in the highest scoring consensus set. Across multiple base LMs, Codet leads to performance gains of \~10% on both HumanEval and MBPP.

== LEVER: Learning to verify language-to-code generation with execution @ni2023lever
LEVER is a technique that builds off existing code LLMs to improve code generation quality. Using an LM, multiple candidate programs are generated and executed. A separate _verifier_ ranks the generated programs based on their execution results together with the original prompts. This technique improves performance by $4-10%$ over the base models.

== Competition Level Code Generation with AlphaCode @li2022competition
AlphaCode is a model built to solve competitive programming problems. A model pre-trained on code and fine-tuned on Codeforces problems is used to generate a large set of sample solutions. These solutions are filtered and clustered on program behavior using their outputs, and a candidate program is picked from each cluster.