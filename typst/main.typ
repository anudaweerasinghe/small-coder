#import "template.typ": *

#show: project.with(
  title: "Small models that write code",
  authors: (
    (name: "Shreeya Khurana", email: "srkhuran@andrew.cmu.edu"),
    (name: "Bradley Teo", email: "bradleyt@andrew.cmu.edu"),
    (name: "Anuda Weerasinghe", email: "wweerasi@andrew.cmu.edu"),
  ),
  date: "May 3, 2024",
)
#set cite(style: "chicago-notes")
#show link: underline

#include "sections/1-introduction.typ"
#include "sections/2-dataset_task.typ"
#include "sections/3-lit_review.typ"
#include "sections/4-approach.typ"
#include "sections/5-experiments.typ"
#include "sections/6-code_overview.typ"
#include "sections/7-timeline.typ"
#include "sections/8-research-log.typ"
#include "sections/9-conclusion.typ"

#pagebreak()
// Add bibliography and create Bibiliography section
#bibliography("bibliography.bib")