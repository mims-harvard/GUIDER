# GUIDER: Joint Controller--Policy optimization

Minimal, anonymous implementation of joint controller supervision and
Monte-Carlo flow-credit GRPO for
guideline decision-flow data. The repository contains only training and
evaluation code; it does not include data or model weights.

Use Python 3.10+ with `torch`, `transformers`, `datasets`, `peft`,
`bitsandbytes`, and `trl` installed.

## Data format

JSONL rows must contain:

```json
{
  "question": "patient vignette and clinical question",
  "options": ["option A", "option B", "option C", "option D"],
  "answer": 0,
  "decision_path": ["clinical state", "decision/action"]
}
```

`answer` may also be an option letter (`A`, `B`, ...). The policy never receives
the oracle path; the path is used only as controller labels and reward data.

## Training

```bash
MODEL_NAME_OR_PATH=/path/to/model TRAIN_FILE=/path/to/train.jsonl \
OUTPUT_DIR=outputs/jointopt bash train.sh
```

## Evaluation

```bash
MODEL_NAME_OR_PATH=/path/to/model ADAPTER_PATH=outputs/jointopt \
EVAL_FILE=/path/to/test.jsonl OUTPUT_FILE=outputs/jointopt/predictions.jsonl \
bash eval.sh
```

Useful optional environment variables include `BATCH_SIZE`, `NUM_GENERATIONS`,
`MAX_COMPLETION_LENGTH`, `CONTROLLER_LOSS_WEIGHT`, `FLOW_REWARD_WEIGHT`,
`CONTROLLER_SAMPLES`, `CONTROLLER_TEMPERATURE`, and `MC_TEMPERATURE`.

For each policy rollout, training samples multiple controller flows, scores
each flow against the annotated path using reference-node coverage (a node is
matched when at least 28% of its non-stopword tokens overlap), applies
temperature-softmax posterior weights, computes policy--controller
non-stopword token Jaccard similarity, and assigns process credit only when
the final answer is correct.

Optional judge prompts for MedGUIDE, AMEGA-LLM, RealMedQA, and CancerGUIDE
are provided in `evaluation_templates.py` and `evaluation_templates.md`.
They are separate from `eval.py`, which does not call external APIs.
