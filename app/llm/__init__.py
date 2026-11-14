"""
Bring-your-own-model LLM layer (plan §15).

    registry.py   — which providers exist, what fields each needs (drives the UI form)
    resolver.py   — pipeline role → configured model + decrypted credentials
    router.py     — generate_chat(): the ONE function the pipeline calls for text
    embeddings.py — embed_texts(): local fastembed or any LiteLLM embedding model
    health.py     — "test connection" for a configured model
    bootstrap.py  — seed a first provider from env vars (headless/CLI use)
"""
