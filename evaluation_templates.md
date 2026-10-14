# Optional evaluation judge templates

`eval.py` performs generation and deterministic scoring without an external
model. The prompt resources for a separate judge pass are in
`evaluation_templates.py`.

- MedGUIDE: 0--5 semantic adherence to the gold decision path.
- AMEGA-LLM: Boolean verdicts for weighted criteria, aggregated to 0--50.
- RealMedQA: 0--5 support by the NICE recommendation plus evidence connection.
- CancerGUIDE: 0--5 clinical equivalence plus evidence connection.

No API client, API key, or judge model is included.
