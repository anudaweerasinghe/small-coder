= Methods
// Description of both (a) your baseline approach and (b) the main methods

== Replicate DeepSeek-Coder 1.5B
We replicate the results obtained in the DeepSeek-Coder paper @guo2024deepseek by testing the DeepSeek-Coder 1.3B model on the HumanEval and MBPP datasets. We use this model because it is of comparable size to our target model, and will represent a state-of-the-art baseline performance. 

== Supervised Finetuning
We applied supervised finetuning to TinyLlama, our small base model which consists of 1.1B parameters. The model was finetuned with a dataset of 18.6k Python code instructions, example inputs, and response code implementations, called `iamtarun/python_code_instructions_18k_alpaca`. 

== Distillation of a Code Generation Model
Subsequently, we distilled from a large code generation model (`Code Llama Python - 13B`) to our fine-tuned small version of TinyLlama. We used TinyLlama because it shares the same tokenizer as the teacher model, enabling a more streamlined implementation of softmax distillation.
// Our base small model will be the 2.7B parameter Phi-2 model.@gunasekar2023textbooks. As this distillation process hasn't been applied before to Code Llama - 70B to a smaller model like Phi-2, this would be a novel application of an existing method. 

== Improve Base Model with Verifier and Generated Tests
With a fine-tuned model in hand, we used the test generation techniques presented in Codet to further improve performance of our 1.1B model on HumanEval.