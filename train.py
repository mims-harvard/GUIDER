#!/usr/bin/env python3
"""GUIDER training for guideline decision flows.

The policy sees only the patient question and options.  ``decision_path`` is
used as a controller target and as an answer-gated process reward; it is never
inserted into the policy prompt.
"""
from __future__ import annotations

import argparse, json, math, re
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, set_seed)
from trl import GRPOConfig, GRPOTrainer


def rows(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        out = [json.loads(x) for x in f if x.strip()]
    if not out: raise ValueError(f"No rows found in {path}")
    return out


def label(i: int) -> str: return chr(65 + i)


def prompt(r: dict[str, Any]) -> str:
    opts = r.get("options", [])
    text = str(r.get("question", r.get("prompt", "")))
    if opts:
        text += "\n" + "\n".join(f"{label(i)}) {x}" for i, x in enumerate(opts))
    return ("You are a clinical decision-flow policy. Given a patient case, "
            "write a concise guideline-style decision flow and finish with "
            "the best answer option.\n\nQuestion:\n" + text +
            "\n\nFormat:\nDecision flow:\n1. <state or criterion>\n"
            "2. <decision or action>\nAnswer: <letter>")


def controller_prompt(r: dict[str, Any]) -> str:
    """Patient-only controller prompt; oracle paths are never shown."""
    question = str(r.get("question", r.get("prompt", "")))
    return ("You are a clinical decision-flow controller. Given a patient "
            "vignette, generate a compact decision flow that will answer the "
            "question.\n\nQuestion: " + question +
            "\n\nGenerate the likely guideline-style decision flow. Use exactly "
            "this format:\n\nDecision flow:\n"
            "1. <short decision state>\n"
            "2. <short decision state>")


def answer_index(r: dict[str, Any]) -> int | None:
    a = r.get("answer", r.get("gold_index"))
    if isinstance(a, int): return a
    if isinstance(a, str):
        m = re.search(r"\b([A-H])\b", a.upper())
        if m: return ord(m.group(1)) - 65
        try: return int(a)
        except ValueError: return None
    return None


def pred_index(text: str, n: int) -> int | None:
    m = re.search(r"(?:final\s+)?answer\s*[:\-]?\s*\(?([A-H])\)?", text, re.I)
    if not m: m = re.search(r"\b([A-H])\b\s*$", text.strip(), re.I)
    if not m: return None
    i = ord(m.group(1).upper()) - 65
    return i if i < n else None


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "if", "in", "is", "it", "of", "on", "or", "patient",
    "patients", "the", "then", "to", "with",
}


def tokens(s: str) -> set[str]:
    """Deterministic PRM tokenizer matching the main implementation."""
    return {x for x in re.findall(r"[a-z0-9]+", str(s).lower())
            if len(x) > 2 and x not in STOPWORDS}


def flow_score(text: str, path: Any) -> float:
    """Fraction of reference decision nodes matched by the proposal.

    A node is matched when at least 28% of its non-stopword tokens occur in
    the candidate flow. This is the reference-node coverage score s_m; it is
    deliberately different from policy/proposal Jaccard similarity.
    """
    if not isinstance(path, list) or not path: return 0.0
    a = tokens(text); nodes = path[:-1]
    if not a or not nodes: return 0.0
    matched = 0
    for node in nodes:
        b = tokens(node)
        coverage = len(a & b) / len(b) if b else 0.0
        matched += coverage >= 0.28
    return matched / len(nodes)


class JointTrainer(GRPOTrainer):
    def __init__(self, *args, controller_weight=1.0, controller_max_length=256,
                 controller_samples=4, controller_temperature=.7, mc_temperature=.2,
                 flow_weight=.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller_weight = float(controller_weight)
        self.controller_max_length = int(controller_max_length)
        self.controller_samples = max(1, int(controller_samples))
        self.controller_temperature = float(controller_temperature)
        self.mc_temperature = max(1e-6, float(mc_temperature))
        self.flow_weight = float(flow_weight)

    @staticmethod
    def _jaccard(a, b):
        aa, bb = tokens(a), tokens(b)
        return len(aa & bb) / len(aa | bb) if aa and bb else 0.0

    def _controller_samples(self, prompt_text):
        tok = self.processing_class
        enc = tok([prompt_text], return_tensors="pt", truncation=True,
                  max_length=2048).to(self.model.device)
        kwargs = dict(max_new_tokens=128, num_return_sequences=self.controller_samples,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        if self.controller_temperature > 0:
            kwargs.update(do_sample=True, temperature=self.controller_temperature, top_p=.9)
        else:
            kwargs.update(do_sample=False, num_return_sequences=1)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad(): generated = self.model.generate(**enc, **kwargs)
        if was_training: self.model.train()
        n = enc["input_ids"].shape[1]
        return [tok.decode(x[n:], skip_special_tokens=True) for x in generated]

    def mc_reward(self, prompts, completions, decision_path, options, answer,
                  controller_prompt=None, **kwargs):
        """GUIDER MC reward: posterior-weighted controller-flow compatibility."""
        cache = {}
        values = []
        controller_prompt = controller_prompt or prompts
        for p, c, path, opts, ans in zip(controller_prompt, completions, decision_path, options, answer):
            opts = opts if isinstance(opts, list) else []
            gold = answer_index({"answer": ans})
            correct = pred_index(str(c), len(opts)) == gold
            if not correct or not isinstance(path, list) or not path:
                values.append(0.0); continue
            samples = cache.setdefault(p, self._controller_samples(p))
            scores = [max(0.0, flow_score(s, path)) for s in samples]
            mx = max(scores) if scores else 0.0
            weights = [math.exp((s - mx) / self.mc_temperature) for s in scores]
            denom = sum(weights) or 1.0
            proc = sum(w * s * self._jaccard(str(c), sample)
                       for w, s, sample in zip(weights, scores, samples)) / denom
            values.append(1.0 + self.flow_weight * proc)
        return values

    def _controller_loss(self, model, batch):
        texts, targets = [], []
        if isinstance(batch, dict):
            # The standard Trainer collator presents columns as lists/tensors.
            n = len(batch.get("question", batch.get("prompt", [])))
            batch = [{k: (v[i].item() if hasattr(v, "ndim") and v.ndim == 1 else v[i])
                      for k, v in batch.items() if hasattr(v, "__len__") and not isinstance(v, str)}
                     for i in range(n)]
        for r in batch:
            if r.get("controller_prompt") and r.get("controller_target"):
                texts.append(str(r["controller_prompt"])); targets.append(str(r["controller_target"]))
                continue
            path = r.get("decision_path", [])
            if not isinstance(path, list) or not path: continue
            texts.append(controller_prompt(r)); targets.append("\nDecision flow:\n" +
                          "\n".join(f"{i+1}. {x}" for i, x in enumerate(path[:-1])))
        if not texts: return None
        tok = self.processing_class
        enc = tok([a + b for a, b in zip(texts, targets)], return_tensors="pt",
                  padding=True, truncation=True, max_length=self.controller_max_length)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        labels = enc["input_ids"].clone()
        for i, text in enumerate(texts):
            n = len(tok(text, add_special_tokens=False).input_ids)
            labels[i, :min(n, labels.shape[1])] = -100
        labels[enc["attention_mask"] == 0] = -100
        return model(**enc, labels=labels).loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        # TRL passes the original dataset rows here before rollout expansion.
        batch = inputs if isinstance(inputs, list) else inputs.get("rows", inputs)
        loss = self._controller_loss(model, batch)
        if loss is not None:
            self.accelerator.backward(self.controller_weight * loss /
                                      (getattr(self, "current_gradient_accumulation_steps", 1) or 1))
        return super().training_step(model, inputs, num_items_in_batch)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--train-file", required=True)
    p.add_argument("--eval-file"); p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=1); p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--generations", type=int, default=4); p.add_argument("--max-prompt", type=int, default=2048)
    p.add_argument("--max-completion", type=int, default=64); p.add_argument("--controller-weight", type=float, default=1.0)
    p.add_argument("--flow-weight", type=float, default=.2); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--controller-samples", type=int, default=4)
    p.add_argument("--controller-temperature", type=float, default=.7)
    p.add_argument("--mc-temperature", type=float, default=.2)
    p.add_argument("--no-quant", action="store_true"); p.add_argument("--no-peft", action="store_true")
    a = p.parse_args(); set_seed(a.seed)
    train_rows = rows(a.train_file)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    quant = None if a.no_quant else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", torch_dtype=torch.bfloat16,
                                                 quantization_config=quant, trust_remote_code=True)
    if not a.no_quant: model = prepare_model_for_kbit_training(model)
    if not a.no_peft:
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=.05,
                         target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                         task_type="CAUSAL_LM"))
    ds = Dataset.from_list([{**r, "prompt": prompt(r), "controller_prompt": controller_prompt(r),
                             "controller_target": "\nDecision flow:\n" + "\n".join(
                                 f"{i+1}. {x}" for i, x in enumerate(r.get("decision_path", [])[:-1]))}
                            for r in train_rows])
    by_prompt = {prompt(r): r for r in train_rows}
    trainer_ref = {}
    def mc_reward(**kw):
        return trainer_ref["trainer"].mc_reward(**kw)
    cfg = GRPOConfig(output_dir=a.output_dir, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        num_generations=a.generations, max_prompt_length=a.max_prompt, max_completion_length=a.max_completion,
        beta=.04, bf16=True, logging_steps=1, save_strategy="epoch", report_to=[],
        remove_unused_columns=False)
    trainer = JointTrainer(model=model, processing_class=tok, reward_funcs=[mc_reward],
        train_dataset=ds, args=cfg, controller_weight=a.controller_weight,
        controller_max_length=256, controller_samples=a.controller_samples,
        controller_temperature=a.controller_temperature, mc_temperature=a.mc_temperature,
        flow_weight=a.flow_weight)
    trainer_ref["trainer"] = trainer
    trainer.train(); trainer.save_model(a.output_dir); tok.save_pretrained(a.output_dir)

if __name__ == "__main__": main()
