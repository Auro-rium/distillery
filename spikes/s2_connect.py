"""S2a: connect to Sandboxes with the Token Factory key; list images; run one command."""

import asyncio
import os
import time

from contree_sdk import Contree


async def main() -> None:
    sdk = Contree(
        token=os.environ["NEBIUS_API_KEY"], base_url="https://api.tokenfactory.nebius.com/sandboxes"
    )
    try:
        t = time.time()
        imgs = await sdk.images()
        print(
            "images ok:",
            len(imgs),
            f"{time.time() - t:.2f}s",
            [getattr(i, "tag", None) for i in imgs][:8],
        )
        t = time.time()
        image = await sdk.images.oci("docker://python:3.12-slim")
        print(
            "import python:3.12-slim ok",
            f"{time.time() - t:.1f}s",
            "uuid=",
            getattr(image, "uuid", None),
        )
        t = time.time()
        r = await image.run(
            shell="python3 -c 'import sqlite3,sys; print(sqlite3.sqlite_version, sys.version.split()[0])'"
        )
        print(
            "run:",
            r.exit_code,
            repr(r.stdout),
            repr(r.stderr[:200] if isinstance(r.stderr, str) else r.stderr),
            f"{time.time() - t:.2f}s",
            "uuid=",
            r.uuid,
        )
    finally:
        await sdk.close() if hasattr(sdk, "close") else None


asyncio.run(main())
