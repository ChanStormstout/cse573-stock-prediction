# Layout and connector contract

## method_overview

| Edge | Source | Destination | Meaning |
|---|---|---|---|
| title_encoder | input | encoder | Data flow |
| title_group | input | group | Data flow |
| embeddings | encoder | pool | Data flow |
| membership | group | pool | Membership dependency |
| metadata | group | meta | Data flow |
| semantic_join | pool | join | Data flow |
| meta_join | meta | join | Data flow |
| price_join | price | join | Data flow |
| classifier | join | lr | Data flow |
| forecast | lr | out | Data flow |

Exact waypoints are in the XML and derived JSON. Forbidden crossing zones: all non-endpoint boxes and text labels.

## grouping_example

| Edge | Source | Destination | Meaning |
|---|---|---|---|
| a | titlesA | meanA | Data flow |
| b | titlesB | meanB | Data flow |
| ma | meanA | mean | Data flow |
| mb | meanB | mean | Data flow |

Exact waypoints are in the XML and derived JSON. Forbidden crossing zones: all non-endpoint boxes and text labels.
