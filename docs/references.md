# References and provenance

The original operational implementation was maintained by Shawn Reddy in the private `Screddyice/shawn-corpus` repository; the consuming MCP bridge lived in `Screddyice/src-os`. `provenance.json` lists the selected original code files, original hashes and exact source revision. This release begins with a clean history and excludes private datasets, transcripts, credentials, health modules and model artifacts.

Original design references cover deliberate agent invocation, result diversification, significance/deduplication, learned reranking, frozen evaluation methods and checkpoint/promotion mistakes. Their operational substance is preserved in the architecture, retrieval, training and operations runbooks. Historical infrastructure addresses, provider accounts, health runbooks and private evaluation records are not public setup instructions.

- [Model Context Protocol documentation](https://modelcontextprotocol.io/docs/getting-started/intro)
- [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
- [pgvector](https://github.com/pgvector/pgvector)
- [Psycopg prepared statements](https://www.psycopg.org/psycopg3/docs/advanced/prepare.html)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Cross-encoder base model](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [Pinned model revision](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a)
- [uv](https://docs.astral.sh/uv/)

Upstream model metadata inspected during extraction reports Apache-2.0. Weights are downloaded separately and retain their upstream terms. Python dependencies retain their respective licenses; they are not relicensed by this repository's MIT source license. No third-party source tree or weights are vendored. The lockfile records exact dependency versions used to verify the extraction.
