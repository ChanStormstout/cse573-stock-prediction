# CMIN-US official protocol recovery

Primary evidence: Luo et al., ACL 2023, Appendix A / Table 3, and the authors' `BigRoddy/CMIN-Dataset` repository at commit `3f571a53b81d8f3ea7b3afb9877bdbdbeae54c97`.

- Universe: top 110 U.S. stocks by market capitalization as described by the paper.
- Data span: 2018-01-01 through 2021-12-31.
- Official train: 2018-01-01 through 2021-04-30.
- Official development: 2021-05-01 through 2021-08-31.
- Official test: 2021-09-01 through 2021-12-31.
- Price source: Yahoo Finance daily history.
- Text source: Yahoo Finance; the paper states experiments use headlines rather than full text.
- Target construction: stock movement classification is described, but the exact released-code label construction was not recovered from the dataset-only repository.
- News timezone: unresolved; repository rows provide date and time without an encoded timezone.

Therefore `CMIN_OFFICIAL` can preserve the paper's dates/universe/headline representation, but exact point-in-time reassignment requires a separately named, reaudited protocol. The earlier survey link to ACL long paper 673 was incorrect; the correct paper is ACL 2023 long paper 679.
