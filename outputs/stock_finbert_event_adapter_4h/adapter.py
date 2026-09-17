"""Multi-task FinBERT sentence adapter and extraction metrics."""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from common import ACTIONS, FINBERT_CACHE, MODEL_ID, TYPES, seed_everything

REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"


def model_kwargs():
    return {
        "revision": REVISION,
        "cache_dir": str(FINBERT_CACHE),
        "local_files_only": True,
        "trust_remote_code": False,
    }


def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, **model_kwargs())


def article_examples(inputs: list[dict], labels: dict[str, dict] | None = None) -> list[dict]:
    result = []
    for article in inputs:
        gold = labels.get(article["id"], {"answer": {"events": []}}) if labels is not None else {"answer": {"events": []}}
        event_by_sentence = {(sentence_id, kind): [] for sentence_id in article["sentences"] for kind in TYPES}
        for event in gold["answer"]["events"]:
            for sentence_id in event["evidence_ids"]:
                event_by_sentence.setdefault((sentence_id, event["kind"]), []).append(event)
        sentence_items = list(article["sentences"].items())
        candidate_ids = set(article.get("candidate_ids", article["sentences"].keys()))
        for sentence_index, (sentence_id, sentence) in enumerate(sentence_items):
            if sentence_id not in candidate_ids:
                continue
            evidence, actions = [], []
            for kind in TYPES:
                events = event_by_sentence.get((sentence_id, kind), [])
                evidence.append(int(bool(events)))
                if events:
                    values = [ACTIONS.index(event["action"]) for event in events]
                    actions.append(max(set(values), key=values.count))
                else:
                    actions.append(-100)
            # The evidence candidate is protected in the first sequence and can
            # never be truncated.  The second sequence contains every complete,
            # numbered sentence in the extracted target-company context.  The
            # tokenizer may shorten only that context at a token boundary; this
            # replaces the former first-600-character and adjacent-sentence
            # approximations while preserving evidence sentence IDs.
            context_rows = [(key, value) for key, value in sentence_items if key != sentence_id]
            context = "\n".join(f"{key}: {value}" for key, value in context_rows) or "NO_OTHER_CONTEXT_SENTENCE"
            result.append(
                {
                    "article_id": article["id"],
                    "symbol": article["symbol"],
                    "split": article.get("split", "inference"),
                    "available_utc": article.get("available_utc"),
                    "record_key": article.get("record_key"),
                    "event_group": article.get("group") or article.get("event_group") or article.get("record_key"),
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "header": f"TARGET={article['symbol']}\nTITLE={article['title']}",
                    "prefix": f"TARGET={article['symbol']}\nTITLE={article['title']}\nEVIDENCE_CANDIDATE {sentence_id}: {sentence}",
                    "context": context,
                    "evidence": evidence,
                    "actions": actions,
                }
            )
    return result


class EncodedDataset(Dataset):
    def __init__(self, examples: list[dict], tok, max_length: int = 512):
        self.examples = []
        self.encodings = []
        for row in examples:
            candidate_intro = f"{row.get('header', row['prefix'].rsplit(chr(10), 1)[0])}\nEVIDENCE_CANDIDATE {row['sentence_id']}: "
            intro_tokens = tok(candidate_intro, add_special_tokens=False)["input_ids"]
            sentence_tokens = tok(row["sentence"], add_special_tokens=False)["input_ids"]
            # Three pair-special tokens plus a small safety margin.  Only
            # inference-corpus web fragments need multiple chunks; all sealed
            # annotation candidates fit in one protected first sequence.
            chunk_budget = max_length - len(intro_tokens) - 6
            if chunk_budget < 32:
                raise ValueError({"candidate_header_too_long": row["article_id"], "sentence_id": row["sentence_id"], "intro_tokens": len(intro_tokens)})
            chunks = [sentence_tokens[offset : offset + chunk_budget] for offset in range(0, len(sentence_tokens), chunk_budget)] or [[]]
            for chunk_index, token_chunk in enumerate(chunks):
                candidate = row["sentence"] if len(chunks) == 1 else tok.decode(token_chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                prefix = candidate_intro + candidate
                meta = dict(row)
                meta.update(model_unit_id=f"{row['sentence_id']}#chunk{chunk_index}", model_chunk_index=chunk_index, model_chunk_count=len(chunks))
                self.examples.append(meta)
                self.encodings.append(
                    tok(
                        prefix,
                        row["context"],
                        truncation="only_second",
                        max_length=max_length,
                        add_special_tokens=True,
                    )
                )
        self.lengths = [len(row["input_ids"]) for row in self.encodings]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return index, self.encodings[index], self.examples[index]["evidence"], self.examples[index]["actions"]


class LengthBucketBatchSampler:
    """Deterministic length buckets reduce padding without changing examples."""

    def __init__(self, dataset, batch_size: int, seed: int = 0, shuffle: bool = False):
        self.lengths = list(dataset.lengths) if hasattr(dataset, "lengths") else [len(row["input_ids"]) for row in dataset.encodings]
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        ordered = sorted(range(len(self.lengths)), key=lambda index: (self.lengths[index], index))
        batches = [ordered[start : start + self.batch_size] for start in range(0, len(ordered), self.batch_size)]
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(batches)
            for batch in batches:
                rng.shuffle(batch)
            self.epoch += 1
        yield from batches


def collator(tok):
    def collate(rows):
        indices, encodings, evidence, actions = zip(*rows)
        batch = tok.pad(list(encodings), padding=True, return_tensors="pt")
        return {
            **batch,
            "indices": torch.tensor(indices),
            "evidence": torch.tensor(evidence, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
        }

    return collate


class CachedTopDataset(Dataset):
    def __init__(self, hidden_rows: list[torch.Tensor], examples: list[dict]):
        self.hidden_rows = hidden_rows
        self.examples = examples
        self.lengths = [len(row) for row in hidden_rows]

    def __len__(self):
        return len(self.hidden_rows)

    def __getitem__(self, index):
        row = self.examples[index]
        return index, self.hidden_rows[index], row["evidence"], row["actions"]


def cached_top_collator(rows):
    indices, hidden, evidence, actions = zip(*rows)
    return {
        "indices": torch.tensor(indices),
        "hidden": pad_sequence(hidden, batch_first=True),
        "attention_mask": pad_sequence([torch.ones(len(row), dtype=torch.long) for row in hidden], batch_first=True),
        "evidence": torch.tensor(evidence, dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.long),
    }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: int = 8):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, value):
        return self.base(value) + self.b(self.a(value)) * self.scale


class EventAdapter(nn.Module):
    def __init__(self, config_name: str):
        super().__init__()
        self.config_name = config_name
        # PyTorch MPS does not implement fused SDPA with training-time dropout.
        # Eager attention is the same BERT attention computation and works for
        # A1/A2 backward passes on Apple Silicon.
        self.encoder = AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager", **model_kwargs())
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.evidence_head = nn.Linear(hidden, len(TYPES))
        self.action_head = nn.Linear(hidden, len(TYPES) * len(ACTIONS))
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if config_name == "A1":
            for layer in self.encoder.encoder.layer[-2:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        elif config_name == "A2":
            for layer in self.encoder.encoder.layer[-2:]:
                layer.attention.self.query = LoRALinear(layer.attention.self.query, rank=4, alpha=8)
                layer.attention.self.value = LoRALinear(layer.attention.self.value, rank=4, alpha=8)
        elif config_name != "A0":
            raise ValueError(config_name)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = self.dropout(output.last_hidden_state[:, 0])
        evidence = self.evidence_head(hidden)
        actions = self.action_head(hidden).reshape(-1, len(TYPES), len(ACTIONS))
        return evidence, actions

    def forward_top(self, hidden, attention_mask):
        extended = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden.dtype)) * torch.finfo(hidden.dtype).min
        for layer in self.encoder.encoder.layer[-2:]:
            hidden = layer(hidden, attention_mask=extended)[0]
        pooled = self.dropout(hidden[:, 0])
        evidence = self.evidence_head(pooled)
        actions = self.action_head(pooled).reshape(-1, len(TYPES), len(ACTIONS))
        return evidence, actions

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config_name == "A0":
            self.encoder.eval()
        elif mode:
            # Only the registered top two layers may use training-time dropout.
            # Embeddings and the frozen lower ten layers stay deterministic and
            # can therefore be cached without changing the trainable function.
            self.encoder.eval()
            for layer in self.encoder.encoder.layer[-2:]:
                layer.train()
        return self


def trainable_state(model: EventAdapter) -> dict:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def load_trainable(model: EventAdapter, state: dict) -> None:
    current = model.state_dict()
    missing = sorted(set(state) - set(current))
    if missing:
        raise ValueError({"missing_checkpoint_keys": missing})
    current.update(state)
    model.load_state_dict(current, strict=True)


def class_weights(examples: list[dict]):
    evidence = np.asarray([row["evidence"] for row in examples], dtype=float)
    positive = evidence.sum(axis=0)
    negative = len(evidence) - positive
    pos_weight = np.clip(negative / np.maximum(positive, 1), 1, 20)
    action_weights = []
    for type_index in range(len(TYPES)):
        counts = np.bincount([row["actions"][type_index] for row in examples if row["actions"][type_index] >= 0], minlength=len(ACTIONS)).astype(float)
        weights = counts.sum() / np.maximum(counts, 1)
        weights = np.clip(weights / max(weights.mean(), 1e-9), 0.25, 4)
        action_weights.append(weights)
    return pos_weight.astype(np.float32), np.asarray(action_weights, dtype=np.float32)


def task_loss(evidence_logits, action_logits, evidence, actions, pos_weight, action_weights):
    evidence_loss = nn.functional.binary_cross_entropy_with_logits(evidence_logits, evidence, pos_weight=pos_weight)
    parts = []
    for type_index in range(len(TYPES)):
        mask = actions[:, type_index] >= 0
        if mask.any():
            parts.append(nn.functional.cross_entropy(action_logits[mask, type_index], actions[mask, type_index], weight=action_weights[type_index]))
    action_loss = torch.stack(parts).mean() if parts else evidence_loss * 0
    return evidence_loss + action_loss, evidence_loss, action_loss


@torch.no_grad()
def infer(model: EventAdapter, dataset: EncodedDataset, tok, device: torch.device, batch_size: int = 32):
    model.eval()
    loader = DataLoader(dataset, batch_sampler=LengthBucketBatchSampler(dataset, batch_size), collate_fn=collator(tok))
    rows = [None] * len(dataset)
    losses = []
    for batch in loader:
        indices = batch.pop("indices").numpy()
        batch.pop("evidence")
        batch.pop("actions")
        values = {key: value.to(device) for key, value in batch.items()}
        evidence, actions = model(**values)
        evidence = torch.sigmoid(evidence).cpu().numpy()
        actions = torch.softmax(actions, dim=-1).cpu().numpy()
        for offset, index in enumerate(indices):
            meta = dataset.examples[int(index)]
            rows[int(index)] = {
                "article_id": meta["article_id"],
                "symbol": meta["symbol"],
                "split": meta["split"],
                "available_utc": meta["available_utc"],
                "record_key": meta["record_key"],
                "event_group": meta["event_group"],
                "sentence_id": meta["sentence_id"],
                "model_unit_id": meta.get("model_unit_id", meta["sentence_id"]),
                "model_chunk_index": meta.get("model_chunk_index", 0),
                "model_chunk_count": meta.get("model_chunk_count", 1),
                "evidence_probability": evidence[offset].tolist(),
                "action_probability": actions[offset].tolist(),
                "gold_evidence": meta["evidence"],
                "gold_actions": meta["actions"],
            }
    return rows


def extraction_metrics(predictions: list[dict], threshold: float, symbol: str = "ALL") -> dict:
    rows = [row for row in predictions if symbol == "ALL" or row["symbol"] == symbol]
    article_ids = sorted({row["article_id"] for row in rows})
    by_article = {article_id: [row for row in rows if row["article_id"] == article_id] for article_id in article_ids}
    gold_event, pred_event = [], []
    gold_type, pred_type = [], []
    gold_action, pred_action = [], []
    gold_evidence, pred_evidence = [], []
    exact = 0
    no_event = false_positive = 0
    for article_id, article_rows in by_article.items():
        gold_facts, pred_facts = set(), set()
        article_gold_types, article_pred_types = [], []
        article_gold_actions, article_pred_actions = [], []
        any_gold = any(any(row["gold_evidence"]) for row in article_rows)
        any_pred = any(any(np.asarray(row["evidence_probability"]) >= threshold) for row in article_rows)
        gold_event.append(int(any_gold))
        pred_event.append(int(any_pred))
        if not any_gold:
            no_event += 1
            false_positive += int(any_pred)
        for type_index, kind in enumerate(TYPES):
            type_gold = any(row["gold_evidence"][type_index] for row in article_rows)
            type_pred = any(row["evidence_probability"][type_index] >= threshold for row in article_rows)
            article_gold_types.append(int(type_gold))
            article_pred_types.append(int(type_pred))
            gold_actions = {row["gold_actions"][type_index] for row in article_rows if row["gold_actions"][type_index] >= 0}
            predicted_rows = [row for row in article_rows if row["evidence_probability"][type_index] >= threshold]
            predicted_actions = {int(np.argmax(row["action_probability"][type_index])) for row in predicted_rows}
            for action in gold_actions:
                gold_facts.add((kind, ACTIONS[action]))
            for action in predicted_actions:
                pred_facts.add((kind, ACTIONS[action]))
            article_gold_actions.append([int(action in gold_actions) for action in range(len(ACTIONS))])
            article_pred_actions.append([int(action in predicted_actions) for action in range(len(ACTIONS))])
        gold_type.append(article_gold_types)
        pred_type.append(article_pred_types)
        gold_action.extend(article_gold_actions)
        pred_action.extend(article_pred_actions)
        exact += int(gold_facts == pred_facts)
        for row in article_rows:
            for type_index in range(len(TYPES)):
                gold_evidence.append(int(row["gold_evidence"][type_index]))
                pred_evidence.append(int(row["evidence_probability"][type_index] >= threshold))
    precision, recall, event_f1, _ = precision_recall_fscore_support(gold_event, pred_event, average="binary", zero_division=0)
    return {
        "symbol": symbol,
        "articles": len(article_ids),
        "positive_articles": int(sum(gold_event)),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(event_f1),
        "type_macro_f1": float(f1_score(np.asarray(gold_type), np.asarray(pred_type), average="macro", zero_division=0)),
        "action_macro_f1": float(f1_score(np.asarray(gold_action), np.asarray(pred_action), average="macro", zero_division=0)),
        "evidence_micro_f1": float(f1_score(gold_evidence, pred_evidence, average="binary", zero_division=0)),
        "no_event_articles": int(no_event),
        "no_event_false_positives": int(false_positive),
        "no_event_false_positive_rate": float(false_positive / no_event) if no_event else None,
        "exact_type_action_set": float(exact / len(article_ids)) if article_ids else None,
        "predicted_positive_articles": int(sum(pred_event)),
    }


def adapter_score(metrics: dict) -> float:
    return float(0.4 * metrics["event_f1"] + 0.2 * metrics["type_macro_f1"] + 0.2 * metrics["action_macro_f1"] + 0.2 * metrics["evidence_micro_f1"])


def optimizer(model: EventAdapter, config_name: str):
    heads, encoder = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (heads if name.startswith("evidence_head") or name.startswith("action_head") else encoder).append(parameter)
    groups = [{"params": heads, "lr": 5e-4}]
    if encoder:
        groups.append({"params": encoder, "lr": 1e-5 if config_name == "A1" else 1e-4})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


@torch.no_grad()
def frozen_features(model: EventAdapter, dataset: EncodedDataset, tok, device: torch.device):
    model.encoder.eval()
    feature_rows, evidence_rows, action_rows = [], [], []
    # Preserve annotation order so A0 cached-head fits are exactly reproducible
    # across the initial and resumed portions of the official run.
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collator(tok))
    for batch in loader:
        batch.pop("indices")
        evidence_rows.append(batch.pop("evidence"))
        action_rows.append(batch.pop("actions"))
        values = {key: value.to(device) for key, value in batch.items()}
        output = model.encoder(**values)
        feature_rows.append(output.last_hidden_state[:, 0].cpu())
    return torch.cat(feature_rows), torch.cat(evidence_rows), torch.cat(action_rows)


@torch.no_grad()
def cached_lower_hidden(model: EventAdapter, dataset: EncodedDataset, tok, device: torch.device) -> CachedTopDataset:
    """Cache deterministic embeddings plus the frozen lower ten BERT layers."""
    model.encoder.eval()
    hidden_rows = [None] * len(dataset)
    loader = DataLoader(dataset, batch_sampler=LengthBucketBatchSampler(dataset, 32), collate_fn=collator(tok))
    for batch in loader:
        indices = batch.pop("indices").numpy()
        batch.pop("evidence")
        batch.pop("actions")
        values = {key: value.to(device) for key, value in batch.items()}
        input_ids = values["input_ids"]
        attention_mask = values["attention_mask"]
        token_type_ids = values.get("token_type_ids")
        hidden = model.encoder.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        extended = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden.dtype)) * torch.finfo(hidden.dtype).min
        for layer in model.encoder.encoder.layer[:-2]:
            hidden = layer(hidden, attention_mask=extended)[0]
        lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
        for offset, index in enumerate(indices):
            hidden_rows[int(index)] = hidden[offset, : int(lengths[offset])].detach().to(dtype=torch.float16).cpu()
    if any(row is None for row in hidden_rows):
        raise AssertionError("incomplete lower-layer cache")
    return CachedTopDataset(hidden_rows, dataset.examples)


def fit_once(config_name: str, seed: int, train_examples: list[dict], eval_examples: list[dict], tok, device: torch.device, max_epochs: int = 12, patience: int = 2, fixed_epochs: int | None = None):
    seed_everything(seed)
    model = EventAdapter(config_name).to(device)
    train_data = EncodedDataset(train_examples, tok)
    eval_data = EncodedDataset(eval_examples, tok) if eval_examples else None
    train_loader = DataLoader(train_data, batch_sampler=LengthBucketBatchSampler(train_data, 16, seed=seed, shuffle=True), collate_fn=collator(tok))
    pos, actions = class_weights(train_examples)
    pos = torch.tensor(pos, device=device)
    actions = torch.tensor(actions, device=device)
    opt = optimizer(model, config_name)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    started = time.time()
    epochs = fixed_epochs or max_epochs
    cached_head_train = cached_head_eval = None
    cached_top_train = cached_top_eval = None
    if config_name == "A0":
        cached_head_train = frozen_features(model, train_data, tok, device)
        if eval_data is not None:
            cached_head_eval = frozen_features(model, eval_data, tok, device)
    else:
        cached_top_train = cached_lower_hidden(model, train_data, tok, device)
        if eval_data is not None:
            cached_top_eval = cached_lower_hidden(model, eval_data, tok, device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = []
        if cached_head_train is not None:
            cached_dataset = torch.utils.data.TensorDataset(*cached_head_train)
            batches = DataLoader(cached_dataset, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed + epoch))
        elif cached_top_train is not None:
            batches = DataLoader(cached_top_train, batch_sampler=LengthBucketBatchSampler(cached_top_train, 16, seed=seed, shuffle=True), collate_fn=cached_top_collator)
        else:
            batches = train_loader
        for batch in batches:
            if cached_head_train is not None:
                features, evidence, gold_actions = batch
                features, evidence, gold_actions = features.to(device), evidence.to(device), gold_actions.to(device)
            elif cached_top_train is not None:
                batch.pop("indices")
                hidden = batch.pop("hidden").to(device=device, dtype=next(model.encoder.encoder.layer[-1].parameters()).dtype)
                attention_mask = batch.pop("attention_mask").to(device)
                evidence = batch.pop("evidence").to(device)
                gold_actions = batch.pop("actions").to(device)
            else:
                batch.pop("indices")
                evidence = batch.pop("evidence").to(device)
                gold_actions = batch.pop("actions").to(device)
                values = {key: value.to(device) for key, value in batch.items()}
            opt.zero_grad(set_to_none=True)
            if cached_head_train is not None:
                hidden = model.dropout(features)
                evidence_logits = model.evidence_head(hidden)
                action_logits = model.action_head(hidden).reshape(-1, len(TYPES), len(ACTIONS))
            elif cached_top_train is not None:
                evidence_logits, action_logits = model.forward_top(hidden, attention_mask)
            else:
                evidence_logits, action_logits = model(**values)
            loss, _, _ = task_loss(evidence_logits, action_logits, evidence, gold_actions, pos, actions)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            train_total.append(float(loss.detach().cpu()))
        validation_loss = None
        if eval_data is not None:
            model.eval()
            values_all = []
            with torch.no_grad():
                if cached_head_eval is not None:
                    eval_batches = DataLoader(torch.utils.data.TensorDataset(*cached_head_eval), batch_size=32, shuffle=False)
                elif cached_top_eval is not None:
                    eval_batches = DataLoader(cached_top_eval, batch_sampler=LengthBucketBatchSampler(cached_top_eval, 32), collate_fn=cached_top_collator)
                else:
                    eval_batches = DataLoader(eval_data, batch_sampler=LengthBucketBatchSampler(eval_data, 32), collate_fn=collator(tok))
                for batch in eval_batches:
                    if cached_head_eval is not None:
                        features, evidence, gold_actions = batch
                        features, evidence, gold_actions = features.to(device), evidence.to(device), gold_actions.to(device)
                        evidence_logits = model.evidence_head(features)
                        action_logits = model.action_head(features).reshape(-1, len(TYPES), len(ACTIONS))
                    elif cached_top_eval is not None:
                        batch.pop("indices")
                        hidden = batch.pop("hidden").to(device=device, dtype=next(model.encoder.encoder.layer[-1].parameters()).dtype)
                        attention_mask = batch.pop("attention_mask").to(device)
                        evidence = batch.pop("evidence").to(device)
                        gold_actions = batch.pop("actions").to(device)
                        evidence_logits, action_logits = model.forward_top(hidden, attention_mask)
                    else:
                        batch.pop("indices")
                        evidence = batch.pop("evidence").to(device)
                        gold_actions = batch.pop("actions").to(device)
                        values = {key: value.to(device) for key, value in batch.items()}
                        evidence_logits, action_logits = model(**values)
                    loss, _, _ = task_loss(evidence_logits, action_logits, evidence, gold_actions, pos, actions)
                    values_all.append(float(loss.cpu()))
            validation_loss = float(np.mean(values_all))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_total)), "validation_loss": validation_loss, "gradient_norm_last": float(gradient_norm.detach().cpu())})
        criterion = validation_loss if validation_loss is not None else history[-1]["train_loss"]
        if fixed_epochs is not None:
            best_state, best_epoch = trainable_state(model), epoch
            continue
        if criterion < best_loss - 1e-5:
            best_loss, best_state, best_epoch, stale = criterion, trainable_state(model), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        best_state = trainable_state(model)
        best_epoch = epochs
    load_trainable(model, best_state)
    eval_predictions = infer(model, eval_data, tok, device) if eval_data is not None else []
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    elapsed = time.time() - started
    evidence = {
        "config": config_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "elapsed_seconds": elapsed,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "class_weights": {"evidence_pos": pos.detach().cpu().tolist(), "actions": actions.detach().cpu().tolist()},
        "history": history,
    }
    return model, best_state, eval_predictions, evidence


def reload_check(config_name: str, state: dict, reference: list[dict], dataset: EncodedDataset, tok, device):
    model = EventAdapter(config_name).to(device)
    load_trainable(model, state)
    repeated = infer(model, dataset, tok, device)
    if len(reference) != len(repeated):
        raise AssertionError("reload row count")
    maximum = 0.0
    for left, right in zip(reference, repeated):
        evidence_difference = np.abs(np.asarray(left["evidence_probability"]) - np.asarray(right["evidence_probability"]))
        action_difference = np.abs(np.asarray(left["action_probability"]) - np.asarray(right["action_probability"]))
        maximum = max(maximum, float(np.max(evidence_difference)), float(np.max(action_difference)))
    if maximum > 1e-6:
        raise AssertionError({"checkpoint_reload_max_abs": maximum})
    return maximum
