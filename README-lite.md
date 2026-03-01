# polibias (lite)

Public dashboard:

- https://politicalbiaswithai.streamlit.app/

This project analyzes political bias in article text using local LLM scoring, then exposes the results in a Streamlit dashboard for transparent inspection.

## What was done

- Scraped article text from configured news sources.
- Ran model scoring on four bias dimensions per article.
- Stored outputs in `data/runs/comparisons/bias_data.csv`.
- Built a read-only dashboard for result exploration and comment inspection.

## Dashboard tabs

- `Explanation`: method summary, tab guide, and exact scoring prompt.
- `Overview`: high-level metrics, source-model heatmap, confidence intervals, model summary table.
- `Model Comparison`: score scatter, model distributions, and sub-bias dimension bars.
- `Source Explorer`: source-level article behavior and article summary stats.
- `Comment Explorer`: raw model comment inspection with source/model/article/run filters.

## Bias dimensions

- `subject_bias`: topic selection leaning.
- `framing_bias`: framing or narrative leaning.
- `treatment_bias`: favorability between political sides.
- `guests_bias`: leaning of quoted/invited voices.

Score convention:

- `-1.0` = left-leaning
- `+1.0` = right-leaning
- `0.0` = neutral/unclear

## Exact prompt

Source of truth: `src/polibias/prompt.md`

```markdown
You are a political-bias scoring tool.

Read the article text and output ONLY valid JSON according to the schema below.

All bias scores are floats in [-1.0, +1.0].

Sign convention (consistent):
-1.0 = left-leaning or favorable to the left
+1.0 = right-leaning or favorable to the right
0.0 = neutral or unclear

Definitions:
1) subject_bias:
   Does the topic selection itself lean left or right?

2) framing_bias:
   Is the framing, tone, or narrative left-leaning or right-leaning?

3) treatment_bias:
   Does the article treat the left or the right more favorably?

4) guests_bias:
   Are quoted speakers or invited voices more left or more right?
   (If no clear political actors, return 0.0.)


=== OUTPUT JSON ONLY ===
Schema:
{
  "subject_bias": <float>,
  "framing_bias": <float>,
  "treatment_bias": <float>,
  "guests_bias": <float>,
  "confidence": <float>,
  "comment": <MAX 2 sentence string>,
}
```
