# Security Policy

## Scope

udfx reads the bytecode of your function. It does not call your function when it compiles.

udfx runs your function only on fallback rows. The engine's own UDF path runs it. For Polars, that
path is `map_batches`. udfx adds no other way to run code.

A wrong result is a correctness bug. Report it with the bug report template. Report these as
security issues:

- A way to make udfx run code that the caller did not supply.
- A crafted function or input that causes memory or code safety problems in udfx.

## Supported versions

udfx is pre-alpha. Only the latest commit on `main` gets fixes.

## Report a vulnerability

Send an email to guanmingchiu@gmail.com. Do not open a public issue. Include:

- A description of the problem.
- Steps or code to reproduce it.
- The udfx, Python, and engine versions.

The maintainer will agree on a disclosure date with you.
