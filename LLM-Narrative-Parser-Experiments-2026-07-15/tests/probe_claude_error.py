#!/usr/bin/env python3
"""Print the full 400 error for the newer Claude models."""
import anthropic

c = anthropic.Anthropic()
for model in ["claude-sonnet-5", "claude-opus-4-8"]:
    try:
        r = c.messages.create(
            model=model, max_tokens=100, temperature=0,
            system="Reply with the JSON {\"ok\": true}",
            messages=[{"role": "user", "content": "go"}])
        print(model, "->", r.content[0].text[:80])
    except Exception as e:
        print(model, "->", str(e)[:400])
