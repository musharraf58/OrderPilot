"""Upload docs/policies/*.md to the DocuMind policy collection. Safe to re-run.

Not part of the test suite and never run in CI. Uses DOCUMIND_API_URL, DOCUMIND_EMAIL,
DOCUMIND_PASSWORD and DOCUMIND_COLLECTION from .env; credentials are never printed.

For each file: if the collection already holds the same file with the same content, it is
left alone. If the content changed (or an earlier upload failed), the new version is
uploaded, and the old one is deleted only after the new one is ready, so the knowledge
base never goes empty. Documents that aren't in docs/policies are reported, not deleted.

    uv run python scripts/seed_policies.py
"""

import asyncio
import hashlib
import sys
import time

from _env import ROOT

from app.clients.documind import Document, DocuMindClient, DocuMindError
from app.core.config import get_settings

POLICY_DIR = ROOT / "docs" / "policies"
READY_TIMEOUT_S = 300


async def wait_until_processed(client: DocuMindClient, document: Document) -> Document:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while document.status in ("pending", "processing"):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{document.original_filename} still {document.status}")
        await asyncio.sleep(2)
        document = await client.get_document(document.id)
    return document


async def main() -> int:
    settings = get_settings()
    if not (settings.DOCUMIND_API_URL and settings.DOCUMIND_EMAIL):
        print("Set DOCUMIND_API_URL, DOCUMIND_EMAIL and DOCUMIND_PASSWORD in .env.")
        return 2
    files = sorted(POLICY_DIR.glob("*.md"))
    print(f"DocuMind collection '{settings.DOCUMIND_COLLECTION}', {len(files)} policy files")

    async with DocuMindClient(
        settings.DOCUMIND_API_URL,
        settings.DOCUMIND_EMAIL,
        settings.DOCUMIND_PASSWORD,
        collection_name=settings.DOCUMIND_COLLECTION,
    ) as client:
        try:
            collection = await client.find_collection(settings.DOCUMIND_COLLECTION)
            if collection is None:
                collection = await client.create_collection(
                    settings.DOCUMIND_COLLECTION, "OrderPilot demo store policies"
                )
                print(f"  created collection {collection.id}")
            existing = await client.list_documents(collection.id)

            failed = False
            for path in files:
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                same_name = [d for d in existing if d.original_filename == path.name]
                keep = next(
                    (d for d in same_name if d.content_hash == digest and d.status != "failed"),
                    None,
                )
                if keep is not None:
                    action = "unchanged"
                else:
                    keep = await client.upload_document(
                        collection.id, path.name, content, "text/markdown"
                    )
                    action = "replaced" if same_name else "uploaded"
                keep = await wait_until_processed(client, keep)
                if keep.status != "ready":
                    print(f"  {path.name}: FAILED ({keep.error_message}); old version kept")
                    failed = True
                    continue
                for old in same_name:
                    if old.id != keep.id:
                        await client.delete_document(old.id)
                print(f"  {path.name}: {action} (ready)")

            others = {d.original_filename for d in existing} - {p.name for p in files}
            for name in sorted(others):
                print(f"  note: {name} is in the collection but not in docs/policies (left as is)")
        except DocuMindError as exc:
            print(f"DocuMind error: {exc.message}")
            return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
