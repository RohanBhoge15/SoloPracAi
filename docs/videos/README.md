# videos/ - the SoloPrac demo recording

One file: **`SoloPrac-demo.webm`** (3:54, 1920x1080, silent, 21 MB).

Recorded against the live stack. One doctor, one patient, start to finish, in
logical order:

| Time | What you see |
|---|---|
| 0:00 | Landing page -> doctor registers -> honest empty dashboard |
| 0:20 | One patient created (email linked for auto-linking later) |
| 0:30 | All five record kinds: lab (HbA1c 8.9, backdated), medication, diagnosis, vital, note |
| 1:00 | Chest X-ray + Brain MRI uploaded and indexed (BiomedCLIP) |
| 1:20 | "lung opacity" pixel search -> chest film 0.396 -> click the hit -> record drawer -> scrollable image viewer |
| 1:45 | Chat summary with citations (real LLM) + malaria abstention with zero citations |
| 2:10 | Correction HbA1c 8.9 -> 7.0: toast, SUPERSEDED/CURRENT pair, chat follows |
| 2:25 | As-of toggle: historical question returns the original 8.9 |
| 2:35 | All four PDFs, each shown on camera: prescription, certificate, invoice, weekly report (real LLM narrative) |
| 3:05 | Patient registers with the linked email -> 4 documents in inbox -> bell -> opens prescription in the viewer |
| 3:25 | Patient uploads profile photo (crop) -> doctor dashboard shows the photo |

**No fallback, verified three ways.** The run finished **39/39 checks
passing**: every chat answer was asserted `degraded=false` on the network
response with no "model unavailable / extractive" chip on screen, and the
weekly narrative was asserted model-written on the API response
(`qwen/qwen3.8-27b` throughout, ~5 LLM calls total).

One deliberate shortcut: Playwright records one file per browser tab, so a
literal two-tab flow would produce two files. Doctor and patient act
sequentially in a single tab - the handoff (issue -> inbox) is identical,
and popup PDF tabs are re-shown in the main tab so they are actually in
the recording.

Re-record with any screen recorder using the chapter list above as the shot
list. Keep the run honest: if a model or upload fails, re-record rather than
editing the video.
