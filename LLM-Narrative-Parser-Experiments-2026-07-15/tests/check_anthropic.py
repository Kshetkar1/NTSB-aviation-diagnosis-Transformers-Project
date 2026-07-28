#!/usr/bin/env python3
"""Verify ANTHROPIC_API_KEY works and list available Claude models."""
import os

print("key present:", bool(os.environ.get("ANTHROPIC_API_KEY")))
import anthropic  # noqa: E402

c = anthropic.Anthropic()
for m in c.models.list(limit=50):
    print(m.id)
