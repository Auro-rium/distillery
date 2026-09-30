"""S1c: exercise OUR LLMClient (routing, structured output, reasoning split, usage) live."""

import asyncio
import os

from pydantic import BaseModel

from distillery.llm import CallRecord, LLMClient, make_openai_client

ROLES = {
    "planner": "nvidia/Nemotron-3-Ultra-550b-a55b",
    "teacher": "nvidia/nemotron-3-super-120b-a12b",
    "triage": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
}


class Out(BaseModel):
    sql: str


class Sink:
    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def record_call(self, record: CallRecord) -> None:
        self.records.append(record)


async def main() -> None:
    sink = Sink()
    client = LLMClient(
        make_openai_client(os.environ["NEBIUS_BASE_URL"], os.environ["NEBIUS_API_KEY"]),
        ROLES,
        sink=sink,
    )
    for role in ROLES:
        r = await client.chat(
            role,  # type: ignore[arg-type]
            [
                {
                    "role": "user",
                    "content": 'Write SQL that counts rows in table accounts. JSON {"sql": "..."}.',
                }
            ],
            purpose="spike",
            json_schema=Out,
            temperature=0,
        )
        print(role, "parsed=", r.parsed, "| has_reasoning=", bool(getattr(r, "reasoning", None)))
    for rec in sink.records:
        print(
            rec.role,
            rec.model.split("/")[-1],
            "in",
            rec.input_tokens,
            "out",
            rec.output_tokens,
            f"{rec.latency_s:.2f}s",
            "ok" if rec.ok else rec.error,
        )


asyncio.run(main())
