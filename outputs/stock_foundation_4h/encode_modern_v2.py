"""Encode the v2 ModernBERT cache with special-token-excluded pooling."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from experiment_v2 import ROOT, W, PRIVATE, OUT, init, inputs, sha, dump
from pooling import special_token_excluded_mean


def main() -> None:
    init()
    _, keys, _, _, _, _ = inputs()
    manifest = ROOT / "outputs" / "stock_integrated_4h" / "prepared" / "embedding_inputs.json"
    titles = dict(json.loads(manifest.read_text())["keys_and_titles"])
    texts = [titles[k] for k in keys]
    model_path = W / "models" / "modern"
    fingerprint = dict(
        manifest=sha(manifest), weights=sha(model_path / "model.safetensors"),
        config=sha(model_path / "config.json"), tokenizer=sha(model_path / "tokenizer.json"),
        revision="31d3d96d5839a03dccd030bea40b77c2649a9a01", max_length=256,
        pooling="special_token_excluded_mean", special_tokens_mask=True,
        dtype="float32", code=sha(Path(__file__)),
    )
    cache = PRIVATE / "modern_cache.json"
    if cache.exists():
        assert json.loads(cache.read_text()) == fingerprint, "cache mismatch"
    else:
        dump(cache, fingerprint)
    if (PRIVATE / "modern_embeddings.npz").exists():
        raise FileExistsError("v2 encoder already complete")
    torch.manual_seed(573); torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tick = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager", reference_compile=False,
    ).eval().to(device)
    model.requires_grad_(False)
    chunkdir = PRIVATE / "modern_chunks"; chunkdir.mkdir(exist_ok=True)
    arrays=[]; tokens=[]; real_counts=[]; calls=0
    for start in range(0, len(keys), 32):
        path = chunkdir / f"{start:05d}.npz"; tx=texts[start:start+32]; kk=keys[start:start+32]
        if path.exists():
            z=np.load(path); assert np.array_equal(z["keys"],kk)
            assert z["pooling"].item()=="special_token_excluded_mean"
            v=z["embeddings"]; lens=z["lengths"]; real=z["real_token_counts"]
        else:
            raw=tokenizer(tx,truncation=False,add_special_tokens=True)["input_ids"]
            lens=np.array([len(x) for x in raw],dtype=np.int32)
            batch=tokenizer(tx,padding=True,truncation=True,max_length=256,
                            return_tensors="pt",return_special_tokens_mask=True).to(device)
            with torch.inference_mode():
                hidden=model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).last_hidden_state
                v=special_token_excluded_mean(hidden,batch["attention_mask"],batch["special_tokens_mask"]).cpu().numpy()
            real=(batch["attention_mask"]*(1-batch["special_tokens_mask"])).sum(dim=1).cpu().numpy().astype(np.int32)
            assert np.isfinite(v).all() and np.all(real>0)
            np.savez_compressed(path,keys=kk,embeddings=v,lengths=lens,real_token_counts=real,
                                pooling=np.array("special_token_excluded_mean")); calls+=1
        arrays.append(v); tokens.extend(lens.tolist()); real_counts.extend(real.tolist())
        if start%320==0: print("modern-v2",start+len(tx),"/",len(keys),"seconds",round(time.monotonic()-tick,1),flush=True)
        if time.monotonic()-tick>7200: raise TimeoutError("2-hour encoder budget reached; chunks preserved")
    emb=np.concatenate(arrays); np.savez_compressed(PRIVATE/"modern_embeddings.npz",keys=keys,embeddings=emb)
    evidence=dict(model="clapAI/Fin-ModernBERT",revision=fingerprint["revision"],device=device,dtype="float32",
                  parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  gradients_present=any(p.grad is not None for p in model.parameters()),encoder_training=False,articles=len(keys),new_batches=calls,
                  truncated_articles=int((np.array(tokens)>256).sum()),mean_real_token_count=float(np.mean(real_counts)),
                  min_real_token_count=int(np.min(real_counts)),seconds=time.monotonic()-tick,
                  embeddings_sha256=sha(PRIVATE/"modern_embeddings.npz"),fingerprint=fingerprint,torch=torch.__version__)
    dump(OUT/"modern_inference.json",evidence); print("COMPLETE",evidence,flush=True)


if __name__=="__main__": main()
