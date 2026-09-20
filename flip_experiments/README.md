# When Chain-of-Thought and Final Answers Disagree

**Answer inference failures in QwQ-32B**

## What problem am I trying to solve?

Chain-of-Thought (CoT) monitoring could be useful for AI safety if a model's written reasoning reveals why it reaches a particular answer. But when the CoT and final answer disagree, it is unclear what that disagreement means.

This project starts from a phenomenon described in ["Chain-of-Thought Reasoning in the Wild Is Not Always Faithful"](https://arxiv.org/abs/2503.08679), called **answer flipping**. Answer flipping is when the model's chain-of-thought appears to support one conclusion, but the final YES/NO answer gives the opposite result. For example, imagine the model's CoT shows: "X was released in 2001 and Y in 1998, so X was released later than Y", but then answers: NO.

There are at least three different explanations:

1. **Hidden disagreement** — the model might secretly favor one answer while its visible reasoning points to another. This is a critical safety issue if CoT is meant to reflect true intent.
2. **Mapping failure** — the step-by-step reasoning might be sound, but the system struggles to map its conceptual deduction into the required discrete label.
3. **Evaluation artifact** — the mismatch could be caused by parsing mistakes or judging errors rather than a genuine defect in the model.

I initially thought some "answer flips" might show a real mismatch between what the model says in its reasoning and the answer it is actually leaning toward, and planned to test this by tracking its answer preference throughout the CoT. But after manually checking the examples, I found no clear cases of hidden disagreement. Most were either evaluation mistakes or failures to translate a correct semantic conclusion into the final Yes/No label, so I changed the project to test that simpler explanation instead.

**Research question:** When CoT and final answers disagree, what kind of failure is actually occurring, and where does it happen?

## Method

### Experiment I: Establish what the "answer flips" actually are

*Scripts: `exp0_extract.py`, `exp0_audit_sheet.py`, `exp0_summarize.py`, `exp0b_reparse.py`*

I used the QwQ-32B traces from ChainScope that were automatically labelled `answer_flipping=YES` — 90 flagged responses. Rather than accepting the automatic label as ground truth, I manually read every trace without viewing the automated evaluation rationale and classified each into four categories:

- **Hidden disagreement** — the CoT clearly supports A, but the final answer is B, with no visible reason for the switch.
- **Mapping failure** — the CoT reaches a clear semantic conclusion, but the final YES/NO label contradicts that conclusion.
- **Evaluation false positive** — there is no genuine flip; the apparent mismatch comes from a parser error, judge mistake, self-correction, or similar evaluation artifact.
- **Ambiguous** — the reasoning never settles clearly enough to classify.

**Result:** of 90 responses labelled as answer flips — 39 were semantic-to-label mapping failures, 45 were evaluation false positives, 6 were ambiguous, and **0 were clear hidden disagreements**.

I then ran a cheaper parser check (`exp0b_reparse.py`): ChainScope sometimes uses the first YES/NO after `</think>`, so I re-extract the last clearly stated answer instead. This changed 12/90 (13%) of the labels.

This experiment was a construct-validity check: before explaining a phenomenon mechanistically, verify that the examples actually instantiate the phenomenon.

### Experiment II: Test whether the real mapping failures are caused by the answer interface

*Scripts: `exp2_data.py`, `exp2_build.py`, `exp2_run.py`, `exp2_score.py`, `exp2_analyze.py`, `exp2_qualitative.py`*

After the audit, the main hypothesis changes. Instead of asking "Where is the hidden answer preference?", I asked: "If the semantic conclusion is already present in the CoT, does changing only how the model is asked to express it remove the inconsistency?"

I took the 39 verified mapping failures and 29 matched non-flip controls. For every example: keep the original question and the model's original CoT, freeze that reasoning so it is not regenerated, remove the final answer, and ask QwQ-32B to express the conclusion using four different interfaces:

- **Original** — same YES/NO question as before.
- **Semantic** — ask directly which entity satisfies the relation, e.g. "Which was released later, X or Y?"
- **A/B** — map the two entities to A and B and ask the model to choose.
- **Polarity** — ask a logically equivalent YES/NO question with the relation reversed, e.g. "Was X later than Y?" becomes "Was Y earlier than X?"

The polarity condition is still YES/NO, so if it repairs the failure, the problem can't simply be "QwQ is bad at binary outputs."

The primary outcome is **P(answer agrees with the frozen CoT conclusion)** — not dataset accuracy — because the question is whether the model can correctly express its own conclusion. Ground-truth accuracy is measured separately.

**Result:** for the mapping-failure examples, consistency with the frozen CoT was:

| Condition | Consistency with frozen CoT |
|---|---|
| Original (YES/NO) | 46.2% |
| Semantic (entity choice) | 78.2% |
| A/B choice | 74.4% |
| Polarity (reversed YES/NO) | 87.2% |

Controls remained high across all conditions.

I bootstrapped over the underlying examples (not treating repeated conditions as independent observations). The semantic-vs-original improvement is 44 percentage points larger for flips than controls, 95% bootstrap CI [+22, +66].

Ground-truth accuracy stays around 55–62% across conditions — the intervention does not make the model factually smarter, it makes the final answer more consistent with the reasoning already present.

**Conclusion:** Experiment II provides behavioral causal evidence that changing the answer interface, while holding the reasoning fixed, changes whether the final answer agrees with that reasoning.

### Experiment III: Where the interface effect appears inside the model

*Scripts: `exp3_resample.py`, `exp3_check_margin.py`, `exp3_tokenize.py`, `exp3_logit_lens.py`, `exp3_dual_probe.py`, `exp3_incremental_probe.py`, `exp3_patch.py`, `exp3_patch_control2.py`, `exp3_patch_followup.py`, `exp3_debug_postthink.py`, `exp3_score.py`, `exp3_run.py`*

Before inspecting activations, I pre-specified a reliability filter: an example qualified for mechanistic analysis only if the original interface failed in all 4 stochastic resamples and the polarity reframe succeeded in all 4. Three of 39 mapping failures met this criterion; one failed to reproduce under the deterministic local setup used for activation analysis, leaving **two case-study examples**.

For these cases I loaded the actual QwQ-32B weights and inspected internal activations. At each transformer layer, at a role-aligned position just before the answer, I computed a signed margin Δ: positive Δ means the model internally favors the answer consistent with its CoT, negative means it favors the inconsistent one.

The original and polarity conditions become strongly separated in the final quarter of the network. For the cleanest case, I performed activation patching: replacing the original failing run's residual-stream activation at one layer with the corresponding activation from the successful polarity run, sweeping across every layer. Late-layer patches change the original run from Δ < 0 to Δ > 0 — the model switches from preferring the CoT-inconsistent answer to the CoT-consistent one.

## Conclusion

For QwQ-32B on the ChainScope task, apparent answer flipping is better explained by evaluation errors and a reproducible failure at the final answer interface than by hidden disagreement between the model's visible reasoning and its decision.

## Setup

```bash
cd flip_experiments
cp .env.example .env   # fill in DEEPINFRA_API_KEY or OPENROUTER_API_KEY
```

Run scripts from the repo root with the project's virtualenv active (see main [README](../README.md) for environment setup). Each script takes `--help` for its exact arguments; `data/` (gitignored) holds intermediate outputs.
