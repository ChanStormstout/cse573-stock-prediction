# Reproduction commands

Set `PYTHON` to the project Python environment, then run:

```bash
${PYTHON} outputs/stock_nextgen_4h/inventory.py --out work/stock-data/nextgen_4h/inventory_v2
${PYTHON} outputs/stock_nextgen_4h/paragraph_inputs.py --out work/stock-data/nextgen_4h/paragraphs_v4
${PYTHON} outputs/stock_nextgen_4h/input_quality.py --inputs work/stock-data/nextgen_4h/paragraphs_v4
${PYTHON} outputs/stock_nextgen_4h/recent_price.py --out work/stock-data/nextgen_4h/price_v1
${PYTHON} outputs/stock_nextgen_4h/llm_choice.py --inputs work/stock-data/nextgen_4h/paragraphs_v4 --out work/stock-data/nextgen_4h/llm_v1 --batch-size 4
${PYTHON} outputs/stock_nextgen_4h/calibrate.py --paragraphs work/stock-data/nextgen_4h/paragraphs_v4 --llm work/stock-data/nextgen_4h/llm_v1 --price work/stock-data/nextgen_4h/price_v1 --out work/stock-data/nextgen_4h/calibration_v1
${PYTHON} outputs/stock_nextgen_4h/fusion.py --price work/stock-data/nextgen_4h/price_v1 --calibration work/stock-data/nextgen_4h/calibration_v1 --out work/stock-data/nextgen_4h/fusion_v1
${PYTHON} outputs/stock_nextgen_4h/cases.py --paragraphs work/stock-data/nextgen_4h/paragraphs_v4 --calibration work/stock-data/nextgen_4h/calibration_v1 --fusion work/stock-data/nextgen_4h/fusion_v1 --out work/stock-data/nextgen_4h/cases_v1
${PYTHON} outputs/stock_nextgen_4h/publish.py --inventory work/stock-data/nextgen_4h/inventory_v2 --paragraphs work/stock-data/nextgen_4h/paragraphs_v4 --llm work/stock-data/nextgen_4h/llm_v1 --price work/stock-data/nextgen_4h/price_v1 --calibration work/stock-data/nextgen_4h/calibration_v1 --fusion work/stock-data/nextgen_4h/fusion_v1 --cases work/stock-data/nextgen_4h/cases_v1 --out outputs/stock_nextgen_4h/runs/v1
${PYTHON} outputs/stock_nextgen_4h/verify.py --work work/stock-data/nextgen_4h --public outputs/stock_nextgen_4h/runs/v1
```

Failed inventory v1, paragraph input versions v1--v3 and unlabeled LLM smoke/benchmark runs are retained locally.
