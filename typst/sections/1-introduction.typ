= Introduction
// Concise overview of the entire executive summary including motivation, proposed method, and results.

Small models that perform well on specific, vertical tasks are very important for most scalable product use-cases. We want to explore building a small model for such a vertical task - Python code generation. We chose Python code generation because it has (1) high quality datasets, (2) pre-trained large models that perform well, and (3) sufficient difficulty to allow for some interesting findings.

We begin by replicating the baseline performance of a state-of-the-art small model (`DeepSeek-Coder 1.3B`)@guo2024deepseek, and the base model we'll be using (`TinyLlama-1.1B`@zhang2024tinyllama), on the HumanEval  @chen2021codex and Mostly Basic Python Programs (MBPP) @austin2021program datasets.

We improve on this base model via (1) supervised fine-tuning, (2) knowledge distillation, and (3) external tool integration. We find that the performance of the small model after using these approaches is significantly improved from the base model.