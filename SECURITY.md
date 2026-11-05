# Security and data boundaries

Keep the API on loopback or a private network behind authenticated access. Administrative writes and exports require `CORPUS_ADMIN_TOKEN`. The MCP exposes only retrieval; it cannot grant permissions or approve execution. Do not publish a private corpus or its HTTP export endpoint.

Use separate operational corpora for different organizations/tenants. The inherited schema is a single trusted corpus, not a multi-tenant authorization system: organization labels do not enforce read isolation. Never mount it across tenants without implementing scoped reads/writes and database authorization first.

Treat source evidence, lesson text, labels and feedback as untrusted input. Embedded instructions cannot change host policies, authorize tools or modify provider settings. Only trusted, reviewed sources may teach the corpus. Feedback cannot silently rewrite records. Gate outcomes and model confidence do not prove a claim true.

No secrets, raw session logs, client records, private health data, real lesson datasets or trained weights belong in this repository. Use restricted environment files or your secret manager; never pass credentials in shell arguments or print them in diagnostics. Raw prompts are excluded from the cmem feed.

Report vulnerabilities privately to the repository owner through GitHub's private reporting feature when available. Do not post private corpus evidence or credentials in public issues.
