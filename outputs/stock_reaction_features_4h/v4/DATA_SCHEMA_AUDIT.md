# Full-corpus schema audit

```json
{
  "index_shape": [
    78055,
    29
  ],
  "index_columns": [
    "archive",
    "member",
    "uuid",
    "url",
    "title",
    "chars",
    "published",
    "crawled",
    "thread_published",
    "language",
    "site",
    "exact_hash",
    "normalized_hash",
    "title_norm",
    "aapl",
    "amzn",
    "apple",
    "amazon",
    "title_aapl",
    "title_amzn",
    "physician",
    "offsets",
    "orgs",
    "published_utc",
    "crawled_utc",
    "thread_published_utc",
    "available_utc",
    "lag_hours",
    "article_key"
  ],
  "raw_json_top_level_keys": [
    "author",
    "crawled",
    "entities",
    "external_links",
    "highlightText",
    "highlightTitle",
    "language",
    "locations",
    "ord_in_thread",
    "organizations",
    "persons",
    "published",
    "text",
    "thread",
    "title",
    "url",
    "uuid"
  ],
  "sample_schema": [
    {
      "top_level_keys": [
        "author",
        "crawled",
        "entities",
        "external_links",
        "highlightText",
        "highlightTitle",
        "language",
        "locations",
        "ord_in_thread",
        "organizations",
        "persons",
        "published",
        "text",
        "thread",
        "title",
        "url",
        "uuid"
      ],
      "text_type": "str",
      "entity_keys": [
        "locations",
        "organizations",
        "persons"
      ]
    },
    {
      "top_level_keys": [
        "author",
        "crawled",
        "entities",
        "external_links",
        "highlightText",
        "highlightTitle",
        "language",
        "locations",
        "ord_in_thread",
        "organizations",
        "persons",
        "published",
        "text",
        "thread",
        "title",
        "url",
        "uuid"
      ],
      "text_type": "str",
      "entity_keys": [
        "locations",
        "organizations",
        "persons"
      ]
    },
    {
      "top_level_keys": [
        "author",
        "crawled",
        "entities",
        "external_links",
        "highlightText",
        "highlightTitle",
        "language",
        "locations",
        "ord_in_thread",
        "organizations",
        "persons",
        "published",
        "text",
        "thread",
        "title",
        "url",
        "uuid"
      ],
      "text_type": "str",
      "entity_keys": [
        "locations",
        "organizations",
        "persons"
      ]
    },
    {
      "top_level_keys": [
        "author",
        "crawled",
        "entities",
        "external_links",
        "highlightText",
        "highlightTitle",
        "language",
        "locations",
        "ord_in_thread",
        "organizations",
        "persons",
        "published",
        "text",
        "thread",
        "title",
        "url",
        "uuid"
      ],
      "text_type": "str",
      "entity_keys": [
        "locations",
        "organizations",
        "persons"
      ]
    },
    {
      "top_level_keys": [
        "author",
        "crawled",
        "entities",
        "external_links",
        "highlightText",
        "highlightTitle",
        "language",
        "locations",
        "ord_in_thread",
        "organizations",
        "persons",
        "published",
        "text",
        "thread",
        "title",
        "url",
        "uuid"
      ],
      "text_type": "str",
      "entity_keys": [
        "locations",
        "organizations",
        "persons"
      ]
    }
  ],
  "text_field": "text",
  "entities_field": "entities",
  "availability_rule": "max(valid published_utc,crawled_utc)",
  "reaction_start_rule": "first bar start >= availability",
  "reaction_maturity_rule": "reaction bar end <= session close and contiguous",
  "price_context_rule": "only bars with bar_end <= availability",
  "pair_unit": "(article_key,target_symbol)",
  "target_pair_key": "article_key|target_symbol",
  "association_review": "deterministic tier-stratified cards; independent human gold not claimed",
  "public_text_policy": "no article text or evidence sentences published"
}
```
