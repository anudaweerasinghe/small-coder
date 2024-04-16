= Approach
// Describe the generative AI methods that you plan to implement and compare. Identify your baseline method and which result(s) you plan to replicate. Be sure to include a clear identification of whether your key contribution will be a novel technical approach, a novel application of an existing method to some task, or a from-scratch reimplementation.

== Replicate DeepSeek-Coder 1.5B
We plan to replicate the results obtained in the DeepSeek-Coder paper @guo2024deepseek by testing the DeepSeek-Coder 1.5B model on the HumanEval and MBPP datasets. We use this model because it is of comparable size to our target model.

== Distillation of a Code Generation Model
Subsequently, we would like to apply model distillation techniques on a large code generation model (such as Code Llama - 70B) to obtain a smaller model with comparable performance. Our base small model will be the 2.7B parameter Phi-2 model.@gunasekar2023textbooks. As this distillation process hasn't been applied before to Code Llama - 70B to a smaller model like Phi-2, this would be a novel application of an existing method. 

== Improve Base Model with Verifier and Generated Tests
With a distilled model in hand, we will implement the test generation techniques presented in Codet and the verification techniques in LEVER. This will also be a novel application of an existing method.