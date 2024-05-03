// Identify specific portions of code that you wrote and explain what those section do in prose. You should either identify a portion of code by including a screenshot of it and placing it in a referenced Figure in an Appendix, or by referring to specific line numbers (e.g. lines 234–270) in the code you upload to Gradescope.

= Code Overview

== Full Fine-Tuning

== Distillation

== CodeT
For CodeT, we wrote a script to generate the required code and test case samples from our chosen model. These scripts are present in the `/codet/` folder. The output `jsonl` files were passed to the CodeT executor in order to perform the dual execution agreement algorithm.