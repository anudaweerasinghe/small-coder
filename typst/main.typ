#import "template.typ": *

#show: project.with(
  title: "Small models that write code",
  authors: (
    (name: "Shreeya Khurana", email: "srkhuran@andrew.cmu.edu"),
    (name: "Bradley Teo", email: "bradleyt@andrew.cmu.edu"),
    (name: "Anuda Weerasinghe", email: "wweerasi@andrew.cmu.edu"),
  ),
  date: "April 4, 2024",
)
#set cite(style: "chicago-notes")
#show link: underline

#include "sections/1-introduction.typ"
#include "sections/2-dataset_task.typ"
#include "sections/3-lit_review.typ"
#include "sections/4-approach.typ"
#include "sections/5-outcomes.typ"
#include "sections/6-plan.typ"

#pagebreak()
// Add bibliography and create Bibiliography section
#bibliography("bibliography.bib")