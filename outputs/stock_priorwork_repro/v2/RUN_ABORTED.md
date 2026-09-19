# Aborted before predictive artifacts

The second invocation exposed a result-table assembly error before it wrote
predictions or metrics. Its manifest and private inputs remain preserved. The
only correction changes list-of-tables assembly to explicit concatenation; the
preregistered methods, grid and protocol are unchanged. The next immutable run
is `v3`.
