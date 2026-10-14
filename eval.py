#!/usr/bin/env python3
"""Deterministic, judge-free evaluation of answer and decision-flow output."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def read(path):
    with open(path) as f: return [json.loads(x) for x in f if x.strip()]
def lab(i): return chr(65+i)
def make_prompt(r):
    q=str(r.get("question",r.get("prompt",""))); opts=r.get("options",[])
    q += "\n"+"\n".join(f"{lab(i)}) {x}" for i,x in enumerate(opts)) if opts else ""
    return "You are a clinical decision-flow policy. Given a patient case, write a concise guideline-style decision flow and finish with the best answer option.\n\nQuestion:\n"+q+"\n\nFormat:\nDecision flow:\n1. <state or criterion>\n2. <decision or action>\nAnswer: <letter>"
def gold(r):
    a=r.get("answer",r.get("gold_index"));
    if isinstance(a,int): return a
    m=re.search(r"\b([A-H])\b",str(a).upper()); return ord(m.group(1))-65 if m else None
def pred(s,n):
    m=re.search(r"(?:final\s+)?answer\s*[:\-]?\s*\(?([A-H])\)?",s,re.I) or re.search(r"\b([A-H])\b\s*$",s,re.I)
    if not m:return None
    i=ord(m.group(1).upper())-65; return i if i<n else None
def score(s,path):
    if not isinstance(path,list) or not path:return 0.0
    tok=lambda x:set(re.findall(r"[a-z0-9]+",str(x).lower()))
    a=tok(s); vals=[]
    for n in path[:-1]:
        b=tok(n); vals.append(len(a&b)/len(a|b) if a and b else 0)
    return sum(vals)/len(vals) if vals else 0
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--adapter")
    p.add_argument("--data",required=True); p.add_argument("--output",required=True); p.add_argument("--max-new-tokens",type=int,default=128); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--no-quant",action="store_true")
    a=p.parse_args(); rows=read(a.data); tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True); tok.padding_side="left"
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    q=None if a.no_quant else BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,device_map="auto",torch_dtype=torch.bfloat16,quantization_config=q,trust_remote_code=True)
    if a.adapter:
        from peft import PeftModel; model=PeftModel.from_pretrained(model,a.adapter)
    out=[]
    for st in range(0,len(rows),a.batch_size):
        batch=rows[st:st+a.batch_size]; enc=tok([make_prompt(r) for r in batch],return_tensors="pt",padding=True,truncation=True,max_length=2048).to(model.device)
        with torch.no_grad(): gen=model.generate(**enc,max_new_tokens=a.max_new_tokens,do_sample=False,pad_token_id=tok.pad_token_id)
        lengths=enc["attention_mask"].sum(dim=1).tolist()
        for r,ids,n in zip(batch,gen,lengths):
            text=tok.decode(ids[int(n):],skip_special_tokens=True); pi=pred(text,len(r.get("options",[])))
            out.append({"question":r.get("question",r.get("prompt","")),"options":r.get("options",[]),"gold_answer":gold(r),"prediction":pi,"correct":pi==gold(r),"decision_path":r.get("decision_path",[]),"completion":text,"flow_score":score(text,r.get("decision_path",[]))})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w") as f:
        for r in out:f.write(json.dumps(r)+"\n")
    print(json.dumps({"n":len(out),"accuracy":sum(r["correct"] for r in out)/len(out),"mean_flow_score":sum(r["flow_score"] for r in out)/len(out),"output":a.output},indent=2))
if __name__=="__main__":main()
