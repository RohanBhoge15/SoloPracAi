# Feasibility Study

> Restored Oct 2026 from report/SoloPrac_Capstone_Report.docx after the
> docs/ folder was lost. No new prose: this is the submitted report
> chapter, extracted mechanically.

## Motivation and scope

1.2 Motivation and scope of the report
Retrieval-augmented generation is now routine in medicine, and benchmarks for it are well established [3], [4]. What those benchmarks share is an assumption that every fact in a record remains true. A benchmark in which that assumption holds cannot detect a system that answers with a value a clinician has already rejected. Recency-based ranking does not repair this either: recency is a prior over relevance, not a constraint on truth, and a survey of temporal information retrieval reaches the same conclusion [5].
The scope of this work is deliberately narrow and is stated as such. It addresses the clinical supersession case only: a value corrected by a clinician who knew it was wrong. It does not claim to be the first temporal-aware benchmark, and temporal question answering over clinical text predates this work. The contribution is a benchmark for the correction case, a retriever that satisfies it, and a shipped product whose retrieval layer is the component being evaluated.

## Problem statement

1.3 Problem statement
Formally, let E be a set of clinical events. Each event e carries a valid time vt(e), a recorded time rt(e), and optionally a supersedes reference s(e) naming the event it corrects. For a present-tense question q about a patient p, the required behaviour is that the ranker places the current version of the relevant fact first, and never returns a superseded value as though it were current. For a question about an instant t, it must instead return the value that was believed correct at t. Stating the first requirement is easy; the difficulty is that the set of superseded events is defined by the shape of the correction graph, not by any property of an individual event. Ranking functions that score events independently cannot express it.

## Hardware

3.6 Hardware description
Table 3  Hardware used

The vision language model runs with all GPU layers offloaded, on CPU with memory mapping. This is a deliberate trade rather than an oversight: the model is about 2.4 GB when quantised, fits in system memory, and keeping it off the device leaves accelerator memory available for the retrieval path, which is the component actually under evaluation. The cost is roughly 140 seconds per image, which is why the vision path is opt-in per request and is never an automatic fallback.

## Software

3.7 Software description
Table 4  Software components

The entire system is brought up by a single command that starts seven services. Readiness is observable rather than assumed: a health endpoint reports service status, whether a language model is configured, and whether the embedding models are warm.
