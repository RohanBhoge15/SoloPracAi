# System Design

> Restored Oct 2026 from report/SoloPrac_Capstone_Report.docx after the
> docs/ folder was lost. No new prose: this is the submitted report
> chapter, extracted mechanically.

3.1 Design principle: validity is derived, never stored
There is no status column on a clinical event. An event is superseded exactly when some other event names it in its supersedes reference:
write:   v2.supersedes_id = v1.id        (a correction is an INSERT)
read:    is v1 superseded?  <=>  exists e : e.supersedes_id = v1.id

Three consequences follow. A correction cannot corrupt history, because corrections append and no code path updates or deletes an event. The status column cannot lie, because no such column exists, which eliminates an entire class of bug. And history becomes a query rather than a stored copy.
The cost is honest and worth stating: derivation happens on read, so the timeline and as-of endpoints do real work per request. At tens of events per patient this is irrelevant; at millions it would require a materialised view with an invalidation strategy. The design is appropriate to the stated scale and not beyond it.

Fig. 1  Block diagram of the SoloPrac system architecture
3.2 Data model

Fig. 2  Database schema: append-only events table and the derived validity view
The schema comprises six tables. Accounts, doctors and patients hold identity and profile; events hold the clinical record; documents record issued artefacts; notifications hold the in-app inbox. The event_validity relation is a view, not a table, so that the interface and reporting never reimplement the chain walk.
Four invariants are enforced by a database check constraint rather than by application convention, because application-level checks hold only until a second write path is added:
1. supersedes must reference an event belonging to the same patient;
1. a correction's valid time may not precede its predecessor's;
1. recorded_at may not precede valid_time;
1. the correction graph may not contain a cycle.
3.3 Three-valued validity
Validity takes three values rather than two: current, superseded, and historical. The third exists because an event superseded today was correctly current in March. Collapsing “was correct then” into “wrong now” would make the as-of view lie.
3.4 Retrieval formulation
A candidate event is scored as the weighted sum of a semantic similarity and a validity weight derived from the supersession graph:
score(event) = a * semantic(query, event) + (1 - a) * validity_weight(event)

Temporal intent is parsed from the query, so that a present-tense question takes the current view while a question about a past instant takes the as-of view. This is why one endpoint answers both correctly rather than requiring the caller to choose.
One implementation detail is worth recording because it produced a spurious result before it was found. Ties are scored by average rank over the method's own scores. An earlier version allowed event identifier ordering to break ties, which silently awarded a token-overlap baseline a fabricated historical accuracy of 99.8 per cent. The dedicated tie tests now exist to prevent that class of bug returning.
3.5 Answer pipeline and honest degradation
The pipeline de-identifies the prompt once, before any provider is contacted, builds context with current facts first and superseded facts labelled, generates through the provider chain, and returns an answer whose citations each carry their own validity state. Real event identifiers are replaced by surrogates in the prompt and restored afterwards, so the model cannot echo an internal identifier. Surrogate substitution is the approach a systematic review of clinical de-identification identifies as dominant in the literature [15], which also reports that leading systems reach headline F1 above ninety-eight per cent while generalising poorly across clinical sub-domains. The system here does not claim that figure: it uses rule-based substitution rather than a trained recogniser, and its scope is narrower than a general de-identification task because only the fields the retriever returns are ever exposed to a provider.
Table 2  Failure modes and the behaviour each produces

The final row is the one that matters for trust. When the model is unavailable the system still answers and says that it did. It never fabricates and never silently substitutes.

Fig. 3  Flow of a chat request through validity-aware retrieval
3.6 Hardware description
Table 3  Hardware used

The vision language model runs with all GPU layers offloaded, on CPU with memory mapping. This is a deliberate trade rather than an oversight: the model is about 2.4 GB when quantised, fits in system memory, and keeping it off the device leaves accelerator memory available for the retrieval path, which is the component actually under evaluation. The cost is roughly 140 seconds per image, which is why the vision path is opt-in per request and is never an automatic fallback.
3.7 Software description
Table 4  Software components

The entire system is brought up by a single command that starts seven services. Readiness is observable rather than assumed: a health endpoint reports service status, whether a language model is configured, and whether the embedding models are warm.
3.8 Benchmark construction
TAP-QA generates synthetic longitudinal patient records with controlled corrections. Four query classes are produced: unchanged facts, corrected facts, historical questions about a superseded value, and contradiction queries in which a note disputes a recorded value. The data are documented following the datasheets discipline [10] and contain no real patient information.
3.9 Summary of the chapter
The architecture is governed by one decision, that validity is derived rather than stored, and the data model, retrieval formulation and degradation behaviour all follow from it. The implementation runs at zero recurring cost on a laptop-class machine and exposes its own readiness and failure states.

Chapter 4
Results and Analysis
