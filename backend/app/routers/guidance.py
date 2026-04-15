import json

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from ..schemas.guidance import GuidanceRequest
from ..services.claude_service import stream_guidance

router = APIRouter(tags=["guidance"])


@router.post("/guidance")
async def guidance(req: GuidanceRequest):
    async def event_generator():
        try:
            async for event in stream_guidance(req):
                event_type = event.pop("type")
                yield {"event": event_type, "data": json.dumps(event)}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_generator())
