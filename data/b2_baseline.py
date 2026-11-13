"""Baseline refusal behaviour per candidate category and language, and the model's own refusal /
compliance onsets (needed to label the refusal EAP task).

For every pool prompt with a text in a language we generate 48 tokens (shared chat format), run
AuditKIT's RefusalDetector, and record the first generated token. Outputs
data/b2/baseline_<cat>.jsonl
and data/b2/baseline_summary.json: refusal rate harmful vs benign per category x language, and the
top-5 first tokens of refusals and of compliances per language."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from aya_circuits.config import load_config
from aya_circuits.model import chat_wrap, load_hf
from neb.judge import refused as judge_refused

HERE = Path(__file__).resolve().parent / "b2"


@torch.no_grad()
def gen_with_first(model, tok, texts, max_new, bs):
    tok.padding_side = "left"
    outs, firsts = [], []
    for i in range(0, len(texts), bs):
        enc = tok(
            texts[i : i + bs], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        g = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
        )
        new = g[:, enc["input_ids"].shape[1] :]
        outs += tok.batch_decode(new, skip_special_tokens=True)
        firsts += new[:, 0].tolist()
    return outs, firsts


def main() -> None:
    cfg = load_config()
    bs, max_new = int(cfg.get("batch_size", 16)), int(cfg["audit"]["max_new_tokens"])
    model, tok = load_hf(cfg)
    summary, onsets = {}, defaultdict(lambda: {"refusal": Counter(), "comply": Counter()})
    for pool in sorted(HERE.glob("pool_*.jsonl")):
        cat = pool.stem[5:]
        rows = [json.loads(line) for line in pool.open()]
        rates = {}
        for lang in cfg["languages"]:
            idx = [i for i, r in enumerate(rows) if lang in r]
            if not idx:
                continue
            texts = [chat_wrap(tok, rows[i][lang], cfg["chat_format"]) for i in idx]
            outs, firsts = gen_with_first(model, tok, texts, max_new, bs)
            for i, o, f in zip(idx, outs, firsts, strict=False):
                refused = judge_refused(rows[i][lang], o, lang)
                rows[i].setdefault("gen", {})[lang] = {
                    "first_token": f,
                    "refused": refused,
                    "preview": o[:160],
                }
                onsets[lang]["refusal" if refused else "comply"][tok.decode([f])] += 1
            for split in ("harmful", "benign"):
                sel = [rows[i]["gen"][lang]["refused"] for i in idx if rows[i]["split"] == split]
                if sel:
                    rates[f"{lang}_{split}"] = {
                        "refusal_rate": round(sum(sel) / len(sel), 3),
                        "n": len(sel),
                    }
            print(cat, lang, {k: v for k, v in rates.items() if k.startswith(lang)}, flush=True)
        # raw rows (with harmful-reply previews) stay local; the repo copy is sanitized (brief §8)
        from data.sanitize import PRIVATE, strip_baseline_row

        PRIVATE.mkdir(parents=True, exist_ok=True)
        (PRIVATE / f"baseline_{cat}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        with (HERE / f"baseline_{cat}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(strip_baseline_row(r), ensure_ascii=False) + "\n")
        summary[cat] = rates
    summary["onsets"] = {
        lang: {k: c.most_common(6) for k, c in v.items()} for lang, v in onsets.items()
    }
    (HERE / "baseline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["onsets"], ensure_ascii=False))


def rejudge() -> None:
    """Recompute `refused`, onsets and the summary from stored previews (no model). Harmful-row
    previews live only in the local raw copy (data/sanitize.py, brief §8); the repo file is
    rewritten sanitized."""
    from transformers import AutoTokenizer

    from data.sanitize import PRIVATE, strip_baseline_row

    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    summary, onsets = {}, defaultdict(lambda: {"refusal": Counter(), "comply": Counter()})
    for pool in sorted(HERE.glob("baseline_*.jsonl")):
        cat = pool.stem[9:]
        raw = PRIVATE / pool.name if (PRIVATE / pool.name).exists() else pool
        rows = [json.loads(line) for line in raw.open()]
        rates = {}
        for lang in cfg["languages"]:
            for r in rows:
                g = r.get("gen", {}).get(lang)
                if g:
                    g["refused"] = judge_refused(r[lang], g["preview"], lang)
                    onsets[lang]["refusal" if g["refused"] else "comply"][
                        tok.decode([g["first_token"]])
                    ] += 1
            for split in ("harmful", "benign"):
                sel = [
                    r["gen"][lang]["refused"]
                    for r in rows
                    if r["split"] == split and lang in r.get("gen", {})
                ]
                if sel:
                    rates[f"{lang}_{split}"] = {
                        "refusal_rate": round(sum(sel) / len(sel), 3),
                        "n": len(sel),
                    }
        if raw != pool:
            raw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        with pool.open("w") as f:
            for r in rows:
                f.write(json.dumps(strip_baseline_row(r), ensure_ascii=False) + "\n")
        summary[cat] = rates
    summary["onsets"] = {
        lang: {k: c.most_common(6) for k, c in v.items()} for lang, v in onsets.items()
    }
    summary["note"] = "judged from 160-char previews with audit/judge.py"
    (HERE / "baseline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "onsets"}, ensure_ascii=False, indent=1
        )
    )
    print(json.dumps(summary["onsets"], ensure_ascii=False))


if __name__ == "__main__":
    (
        rejudge() if sys.argv.pop(sys.argv.index("--rejudge")) else None
    ) if "--rejudge" in sys.argv else main()
