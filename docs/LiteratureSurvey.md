# Literature Survey

> Restored Oct 2026 from report/SoloPrac_Capstone_Report.docx after the
> docs/ folder was lost. No new prose: this is the submitted report
> chapter, extracted mechanically.

2.1 Temporal data: the vocabulary the design adopts
Snodgrass and Ahn formalised temporal databases, introducing the separation of valid time from transaction time [1]. Jensen and colleagues produced the consensus glossary that fixed the terminology still in use, in which a single fact carries both time dimensions and a corrected fact is not deleted but ceases to be current beyond a point [2].
This distinction determined the data model adopted here. The events table carries valid_time and recorded_at as separate columns, and a correction is an insert. The consequence is that the question “what did we believe at date X?” is answered by filtering on recorded_at and re-deriving validity, with no historical status rows to keep consistent.
2.2 Temporal information retrieval: why ranking is insufficient
Campos and colleagues surveyed temporal information retrieval, covering time-aware ranking, time-sensitive query intent and temporal collections [5]. The finding most relevant here is that recency functions as a prior rather than a constraint: a newer document is more likely relevant, which is not the same as an older document being false.
The probabilistic ranking framework of Robertson and Zaragoza provides the lexical family [6], while Karpukhin and colleagues established dense passage retrieval [7] and Cormack and colleagues the standard fusion method [8]. Together these define the baselines evaluated in Chapter 4. None of them can express supersession, because supersession is a relation between records and not a property of any one record.
This limitation was tested rather than assumed. A 35-point sweep over decay weightings and half-lives was run; no setting reached non-zero historical accuracy. The result appears in Table 8.
2.3 Medical retrieval: the encoder and the benchmark landscape
MedCPT, contrastively pretrained on large-scale PubMed search logs, provides a query encoder and an article encoder in a shared 768-dimensional space [9]. It is the text encoder of this system, chosen partly on feasibility grounds since it runs locally and incurs no per-query cost.
Benchmarking work by Xiong and colleagues and by Chen and colleagues established that RAG evaluation in medicine is routine [3], [4]. Their benchmarks measure whether the right passage is retrieved from a record whose statements remain true throughout, which is precisely the condition under which a system can answer with a rejected value and still appear correct. Documentation discipline follows Gebru and colleagues [10], and provenance terms follow the W3C PROV-DM recommendation [11].
2.4 Statistical treatment
Because the same questions are evaluated under two rankers, the paired proportions test of McNemar applies [12]. It is applied per seed rather than pooled, since pooling would obscure that one seed is not significant alone. Because queries within a patient are not independent, a patient-clustered bootstrap is reported alongside the unclustered interval.
2.5 Gap identified
The literature supports temporal data modelling, gives principled ranking methods, and benchmarks medical RAG, but the three are not joined at the point that matters clinically. No benchmark in the reviewed work tests whether a system avoids returning a value a clinician has explicitly corrected, and the distinguishing feature of such a value is that it was rejected deliberately rather than merely aged. The gap is narrow but specific. Retrieval-augmented generation was formulated for knowledge that is stable over the period being questioned [13], and its evaluation assumes the retrieved passage continues to hold; nothing in the formulation distinguishes a passage that is still true from one that was true and has since been replaced. Recent clinical benchmarks move closer to the problem: MEDEC targets the detection and correction of errors within a note [14], which establishes that the clinical community considers erroneous entries worth detecting, but it asks whether the text in front of the reader is wrong. This work asks a different question, namely whether the system will treat a correctly recorded value as current once a clinician has retracted it, and whether it can still return that value when the question genuinely refers to the past. Table 1 summarises the approaches considered.
Table 1  Comparison of approaches to clinical record correction

Two conclusions follow from the survey. First, that the correction case is genuinely unaddressed rather than merely unpopular. Second, and less comfortably, that the weighting scheme this project originally intended to claim as its contribution is not the component that carries the result. That observation shaped the evaluation reported in Chapter 4.

Chapter 3
Methodology and Implementation
Introduction
