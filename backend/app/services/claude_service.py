"""Claude streaming wrapper. Uses prompt caching on the system prompt to cut latency/cost."""
from typing import AsyncIterator

from anthropic import AsyncAnthropic

from ..config import settings
from ..prompts.guidance_template import SYSTEM_PROMPT, build_user_message
from ..schemas.guidance import GuidanceRequest

_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

SECTION_HEADERS = {
    "## Diet": "diet",
    "## Hydration": "hydration",
    "## Exercise": "exercise",
    "## OTC Medicine": "otc_medicine",
    "## Red Flags": "red_flags",
    "## Disclaimer": "disclaimer",
}


async def stream_guidance(req: GuidanceRequest) -> AsyncIterator[dict]:
    """Yields events: {'type': 'section_start'|'token'|'section_end'|'done', ...}"""
    user_msg = build_user_message(req)

    async with _client.messages.stream(
        model=settings.claude_model,
        max_tokens=1500,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        current_section: str | None = None
        buffer = ""
        async for text in stream.text_stream:
            buffer += text
            # Detect section transitions on newlines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                stripped = line.strip()
                if stripped in SECTION_HEADERS:
                    if current_section is not None:
                        yield {"type": "section_end", "section": current_section}
                    current_section = SECTION_HEADERS[stripped]
                    yield {"type": "section_start", "section": current_section}
                else:
                    if current_section and stripped:
                        yield {"type": "token", "text": line + "\n"}
        # Flush remaining buffer
        if current_section and buffer.strip():
            yield {"type": "token", "text": buffer}
        if current_section:
            yield {"type": "section_end", "section": current_section}
        yield {"type": "done"}
