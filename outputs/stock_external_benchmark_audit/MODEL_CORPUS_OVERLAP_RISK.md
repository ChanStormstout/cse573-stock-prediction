# Model-corpus overlap risk

| Model | Published corpus facts | Benchmark risk |
|---|---|---|
| ProsusAI FinBERT | Financial PhraseBank fine-tuning and financial-domain language-model adaptation are documented; exact web-document membership is not published | Exact overlap with candidate news is UNKNOWN |
| Fin-ModernBERT | Model card lists `sabareesh88/FNSPID_nasdaq` among continued-pretraining sources; approximately 50m records were deduplicated to approximately 20m | FNSPID evaluation is not automatically unseen external text; no outcome-label leakage is claimed |
| Qwen3 small reader | Official sources describe large multilingual web, partner, labeled and synthetic corpora but do not publish document-level membership or a precise benchmark exclusion list | Exact overlap with every candidate corpus is UNKNOWN; treat as a reader with locked human evaluation, not proof of unseen-company knowledge |

Model release date or a broad training cutoff cannot prove absence of earlier benchmark text. Report representation results with this risk and keep lexical/rule controls.
