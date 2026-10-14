MEDGUIDE_JUDGE_TEMPLATE = r"""Evaluate whether the model-generated clinical decision flow adheres to the gold MedGUIDE guideline path for this case.

Judge semantic clinical equivalence and ordering, not exact wording. Do not reward generic guideline-like statements unless they correspond to a gold path step. Do not grade MCQ answer correctness as the main criterion; focus on whether the decision flow uses the right clinical criteria and reaches a path compatible with the gold trajectory.

5 = all key gold decision states are captured in a clinically equivalent order, with no important contradictions.
4 = most key states are captured; only minor omissions or mild ambiguity.
3 = partially aligned but misses an important criterion, branch, or action.
2 = weak or mostly generic alignment; little specific path adherence.
1 = mostly wrong path or clinically contradicted by the gold path.
0 = empty or unusable decision flow.

Return JSON only: {"score": <integer 0-5>}"""

AMEGA_JUDGE_TEMPLATE = r"""Evaluate whether a model answer satisfies each AMEGA scoring criterion.

Judge only the model answer against the criterion and the case/question.
Return true only when the answer clearly satisfies the criterion. Return false
for missing, contradicted, or ambiguous information. Do not award credit for
unsafe or clinically inappropriate statements.

Return exactly: {"verdicts": [true, false, ...]}
Each true verdict receives the criterion's predefined weight. Report:
AMEGA_50 = 50 * (sum of satisfied criterion weights) / (sum of all criterion weights)."""

REALMEDQA_JUDGE_TEMPLATE = r"""Evaluate whether a model-generated clinical decision/support flow is supported by a NICE guideline recommendation. Use only the clinical question, reference recommendation, and model output.

5 = fully faithful; 4 = mostly faithful with minor omissions; 3 = partially supported or incomplete; 2 = weakly related with important missing/unsupported components; 1 = mostly unsupported or misleading; 0 = contradicted, unsafe, or unrelated.

Also judge whether the flow connects the question to the recommendation. Return JSON only: {"score": <integer 0-5>, "connects_evidence": <true|false>}"""

CANCERGUIDE_JUDGE_TEMPLATE = r"""Evaluate an NSCLC treatment recommendation against a reference CancerGUIDE recommendation. Use only the patient profile, reference recommendation, and model output.

5 = fully equivalent and faithful; 4 = mostly faithful with minor omissions; 3 = partially supported or incomplete; 2 = weakly related with important missing/unsupported components; 1 = mostly unsupported or misleading; 0 = contradicted, unsafe, or unrelated.

Also judge whether the flow connects the question to the recommendation. Return JSON only: {"score": <integer 0-5>, "connects_evidence": <true|false>}"""

TEMPLATES = {"medguide": MEDGUIDE_JUDGE_TEMPLATE, "amega": AMEGA_JUDGE_TEMPLATE,
             "realmedqa": REALMEDQA_JUDGE_TEMPLATE, "cancerguide": CANCERGUIDE_JUDGE_TEMPLATE}
