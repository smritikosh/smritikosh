# Security

## Supported versions

Smritikosh is alpha. Only the latest release on
[PyPI](https://pypi.org/project/smritikosh/) receives fixes; there are no backports to
earlier versions.

## Reporting a vulnerability

Report privately through
[GitHub Security Advisories](https://github.com/smritikosh/smritikosh/security/advisories/new).
That channel is visible only to maintainers and lets us prepare a fix before anything is
public.

Do not open a public issue for a vulnerability.

What helps:

- What an attacker gains, and what they need to already have
- The smallest reproduction you can manage: a repository shape, a file, or a query
- Version (`smritikosh --version`), Python version, and operating system

Expect an acknowledgement within a week. We will tell you when a fix ships and credit you in
the advisory unless you would rather stay anonymous.

## Scope

Smritikosh runs locally, reads the paths you point it at, and writes one DuckDB file. It
makes no network calls except to download the embedding model on first use. Findings that
matter most:

- Code execution triggered by indexing an untrusted repository or document
- A path outside the indexed tree being read or written
- Source content leaving the machine
- Anything that lets a crafted file corrupt or poison another project's index

Out of scope: vulnerabilities in dependencies with no exploit path through Smritikosh
(report those upstream), and the deliberate behaviour that indexing reads the files you ask
it to read.
