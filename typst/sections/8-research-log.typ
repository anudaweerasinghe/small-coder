// Section 8: Research Log. Explain the meandering path that you took to arrive at the work that this summary represents. Try to explain what the key challenges were and how (or if) you overcame them. This section is your opportunity to showcase the work that you did and is arguably the most important part of this document. For example, if an important line of your results table is empty, you can use this part of the document to explain why it wasn’t easy to fill in. If your plan deviated from your proposal, this is your chance to describe the work that you did to realize your plan needed to change

= Research Log

== Full Fine-Tuning

== Distillation


== CodeT
One of the challenges with CodeT was the long time it took to generate the required code and test case samples. For instance, generating $n=60$ samples for both took several hours each. Thus, only limited experimentation could be done with tuning the temperature.

Some of our results with varying $n$ and temperature $T$ can be found below:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 5pt,
    align: horizon,
    table.header(
      [$n$], [$T$], [pass@1], [pass@2], [pass@10]
    ),
    "20",
    "1.0",
    "14.0%",
    "18.0%",
    "24.8%",
    "60",
    "0.2",
    "13.4%",
    "15.9%",
    "20.0%",
    "60",
    "0.5",
    "15.8%",
    "20.4%",
    "26.6%"
  ),
  caption: "A comparison of pass@k performance for varying numbers of samples and temperature."
)