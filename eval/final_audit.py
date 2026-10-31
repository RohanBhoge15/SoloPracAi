"""Final audit: every number the paper asserts, checked against the artifacts.

Deliberately independent of the harness's own markdown renderer - this reads
report.json / ablation.json / generation_k1.json directly and compares to
the claims transcribed from the .tex by hand.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")

def load(p):
    with open(os.path.join(RES, p), encoding="utf-8") as fh:
        return json.load(fh)

s1 = load("tapqa/report.json")
s2 = load("tapqa_seed20260714/report.json")
ab = load("ablation.json")
gk1 = load("generation_k1.json")

fails, checks = [], 0

def ck(name, claimed, actual, tol=1e-9):
    global checks
    checks += 1
    ok = actual is not None and abs(claimed - actual) <= tol
    if not ok:
        fails.append("%s: paper %.4f vs artifact %.4f" % (name, claimed, actual))

def m(rep, key, field):
    for x in rep["methods"]:
        if x["key"] == key:
            return x[field]
    return None

# ---- Table I : SAR s1/s2/avg, Hist s1/s2, Corr s1/s2 -------------------
T1 = {
    "random":        (17.0, 22.5, 19.8,  9.2,  7.2,  4.5,  5.5),
    "lexical":       (100.0, 100.0, 100.0, 0.0, 0.0,  0.0,  0.0),
    "dense":         (49.5, 53.5, 51.5, 45.8, 53.0, 50.5, 46.0),
    "rrf":           (69.5, 73.0, 71.3, 55.0, 60.5,  1.5,  1.5),
    "valid_time":    (0.0, 0.0, 0.0,     0.0,  0.0, 100.0, 99.5),
    "txn_time":      (21.5, 24.0, 22.8, 21.2, 23.2, 78.5, 75.5),
    "latest_n":      (0.0, 0.0, 0.0,     0.0,  0.0, 10.5, 11.5),
    "bitemporal":    (7.0, 11.5, 9.3,    6.5,  7.5, 87.5, 83.0),
    "ours":          (0.0, 0.0, 0.0,    97.8, 97.8, 98.5, 96.0),
}
print("TABLE I")
for key, (sar1, sar2, saravg, h1, h2, c1, c2) in T1.items():
    ck(key + " SAR s1", sar1, 100 * m(s1, key, "stale_answer_rate"), 0.05)
    ck(key + " SAR s2", sar2, 100 * m(s2, key, "stale_answer_rate"), 0.05)
    ck(key + " SAR avg", saravg,
       (100 * m(s1, key, "stale_answer_rate") + 100 * m(s2, key, "stale_answer_rate")) / 2, 0.05)
    ck(key + " Hist s1", h1, 100 * m(s1, key, "historical_accuracy"), 0.05)
    ck(key + " Hist s2", h2, 100 * m(s2, key, "historical_accuracy"), 0.05)
    ck(key + " Corr s1", c1, 100 * m(s1, key, "corrected_accuracy"), 0.05)
    ck(key + " Corr s2", c2, 100 * m(s2, key, "corrected_accuracy"), 0.05)
print("  %d cells checked" % (len(T1) * 7))

# ---- Table II : dating counterfactual ----------------------------------
print("TABLE II")
cf = ab["counterfactual_same_valid_time"]
T2 = {
    "valid_time_only_default": (0.0, 100.0, 0.0),
    "valid_time_only_same_vt": (100.0, 0.0, 0.0),
    "bitemporal_default": (7.0, 87.5, 6.5),
    "bitemporal_same_vt": (70.0, 24.5, 20.0),
}
for k, (sar, corr, hist) in T2.items():
    ck(k + " SAR", sar, 100 * cf[k]["stale_answer_rate"], 0.05)
    ck(k + " Corr", corr, 100 * cf[k]["corrected_accuracy"], 0.05)
    ck(k + " Hist", hist, 100 * cf[k]["historical_accuracy"], 0.05)
ck("ours default SAR", 0.0, 100 * cf["ours_default"]["stale_answer_rate"], 0.05)
ck("ours default Corr", 98.5, 100 * cf["ours_default"]["corrected_accuracy"], 0.05)
ck("ours same_vt SAR", 0.0, 100 * cf["ours_same_vt"]["stale_answer_rate"], 0.05)
print("  counterfactual rows checked")
print("  NOTE ours same_vt hist is %.1f%% (paper must NOT claim invariance)"
      % (100 * cf["ours_same_vt"]["historical_accuracy"]))

# ---- VI-B sweep --------------------------------------------------------
print("VI-B sweep")
for tau, lo, hi in [(7, 64.5, 70.0)]:
    vals = [100 * x["stale_answer_rate"] for x in s1["methods"]
            if x["sweep_of"] == "decay" and x["params"]["tau"] == 7.0]
    ck("tau=7 min", lo, min(vals), 0.05)
    ck("tau=7 max", hi, max(vals), 0.05)
    vals2 = [100 * x["stale_answer_rate"] for x in s2["methods"]
             if x["sweep_of"] == "decay" and x["params"]["tau"] == 7.0]
    ck("tau=7 s2 min", 66.0, min(vals2), 0.05)
    ck("tau=7 s2 max", 71.5, max(vals2), 0.05)
big = [x for x in s1["methods"] + s2["methods"] if x["sweep_of"] == "decay"]
ck("grid points", 70, len(big), 0)
nz = [x["key"] for x in big if x["historical_accuracy"] > 0]
ck("points with hist>0", 0, len(nz), 0)
print("  %d grid points, %d with hist>0" % (len(big), len(nz)))

# ---- VI-D k=1 ----------------------------------------------------------
print("VI-D k=1")
c = gk1["corrected"]
ck("k1 ours stale", 0.0, 100 * c["ours_stale_rate"], 0.05)
ck("k1 baseline stale", 45.0, 100 * c["baseline_stale_rate"], 0.05)
ck("k1 discordant", 18, c["mcnemar"]["discordant"], 0)
ck("k1 p", 8e-06, c["mcnemar"]["p"], 1e-9)
ck("k1 ours acc", 100.0, 100 * c["ours_accuracy"], 0.05)
print("  control ours %.3f baseline %.3f" % (
    gk1["unchanged_control"]["ours_accuracy"],
    gk1["unchanged_control"]["baseline_accuracy"]))

# ---- significance ------------------------------------------------------
print("SIGNIFICANCE")
sig = {(s["baseline"], s["historical_bootstrap"]["point"], s["historical_bootstrap"]["low"],
        s["historical_bootstrap"]["high"], s["historical_bootstrap"]["significant"])
       for s in s1["significance"]}
print("  s1: %d comparisons, %d significant on historical"
      % (len(s1["significance"]), sum(1 for s in s1["significance"] if s["historical_bootstrap"]["significant"])))
print("  s2: %d comparisons, %d significant on historical"
      % (len(s2["significance"]), sum(1 for s in s2["significance"] if s["historical_bootstrap"]["significant"])))
print("  bitemporal s1 p=%.3g (paper: 2.7e-05)" % [s["safe_answer_mcnemar"]["p_value"]
      for s in s1["significance"] if s["baseline"] == "bitemporal"][0])
print("  bitemporal s2 p=%.3g (paper: 4.2e-05)" % [s["safe_answer_mcnemar"]["p_value"]
      for s in s2["significance"] if s["baseline"] == "bitemporal"][0])
print("  dense s1 p=%.3g (paper: 8.0e-27)" % [s["safe_answer_mcnemar"]["p_value"]
      for s in s1["significance"] if s["baseline"] == "dense"][0])
print("  valid_time joint p: s1=%.2f (paper: 0.25)  s2=%.4f (paper: 0.016)"
      % ([s["safe_answer_mcnemar"]["p_value"] for s in s1["significance"] if s["baseline"] == "valid_time"][0],
         [s["safe_answer_mcnemar"]["p_value"] for s in s2["significance"] if s["baseline"] == "valid_time"][0]))
for s in s2["significance"]:
    if s["baseline"] == "bitemporal":
        h = s["historical_bootstrap"]
        print("  bitemporal hist s2: %+.4f CI[%+.4f,%+.4f] (paper: +0.9025 CI[+0.870,+0.933])" % (h["point"], h["low"], h["high"]))
    if s["baseline"] == "dense":
        print("  dense s2 p=%.3g (paper: 2.0e-25)" % s["safe_answer_mcnemar"]["p_value"])

print()
print("=" * 60)
print("%d numeric assertions checked, %d mismatches" % (checks, len(fails)))
for f in fails:
    print("  MISMATCH", f)