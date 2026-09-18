# TabPFN v2.5 audit

The bundled runtime is TabPFN 6.3.0. Its metadata identifies the default classifier checkpoint as `Prior-Labs/tabpfn_2_5` and says it is fine-tuned on real data. The checkpoint archive root is `tabpfn-v2.5-classifier-v2.5_real-large-samples-and-features`, which is consistent with that metadata. Therefore the historical v1 result must not be called synthetic-only.

- checkpoint SHA-256: `5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12`
- checkpoint size: `42935499` bytes
- v1 configuration: `n_estimators=1` (reduced-compute probe)
- v2 configuration: `n_estimators=8` (library default ensemble)

This audit verifies local provenance and configuration; it does not establish that the model is suitable for stock direction or reproduce the pretraining data.
