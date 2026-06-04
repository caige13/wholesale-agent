"""Marks ``test`` as a regular package so ``from test.fakes import ...`` resolves
to this directory on every Python.

Without this file ``test`` is only a *namespace* package, and CPython's standard
library ships a regular package of the same name (``Lib/test``). A regular
package always wins over a namespace one, so on a stock interpreter (or a slim
Docker image) ``import test`` would bind to the stdlib and shared test helpers
would fail to import. It only worked under uv because uv's standalone Python
omits the stdlib ``test`` package. This file removes that fragility.
"""