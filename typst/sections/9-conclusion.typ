// Summary of your main findings and considerations for future work.

= Conclusion

There is great promise in the use of small, local models for code generation.

Although we were unable to reach the state-of-the-art performance of `DeepSeek-Coder-1.3B`, we gained a better understanding of the approaches used to boost the performance of models intrinsically (fine tuning and distillation) and extrinsically (testcase generation).

There are a number of future directions we would like to take this research given more time.

== Distillation with more Compute & Data
We limited ourselves to distilling from a 7B quantized model due to memory issues. Distillation would work better with a larger teacher model. 

Training for longer with more data is also recommended since the lower temperature distillation runs, which gave the best results, are yet to converge.

We could also explore the possibility of synthetic data as the domain of code generation is especially suited for this. Data that contains code and information about its execution traces could be useful in helping the model learn how code executes.

== Speculative Decoding
Techniques like speculative decoding can be applied to obtain faster results from larger models given the poor results obtained with small models. 

== Attention Distillation
Softmax distillation didn't do as well as we hoped because the softmax distribution is over the entire vocabulary of 32k different tokens. More advanced distillation methods like attention distillation may yield better results.

== IDE Integration
Benchmarks like HumanEval/MBPP are not entirely representative of model performance. Integrating the small model with an IDE, and running it on device, which was our original motivation for this research, can allow for better real-world user testing. 