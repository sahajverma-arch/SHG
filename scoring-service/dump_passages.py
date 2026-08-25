"""Write every passage the app can serve to passages.txt, for pre-rendering.

    python dump_passages.py            # -> passages.txt
    C:\\if5-venv\\Scripts\\python prerender_tts.py --file passages.txt

The list has to come from the database rather than from `supabase/seed.sql`,
because the seed is what was inserted once and the table is what a child is
actually shown. A clip rendered for a passage nobody serves is invisible; worse,
a passage edited in the table silently stops matching its clip, since
`tts_cache.prerender_key` hashes the text.

Two things about reading it back, both of which look like an empty table:

  * `passages_read` requires `auth.role() = 'authenticated'`. The publishable
    key alone is the `anon` role, and RLS *filters* rather than refusing — so
    the request returns 200 with `[]` and nothing tells you why. The web app
    gets past this with `signInAnonymously()`; this does the same call.
  * The table has no `title` column. Selecting one is a 400, not a warning.

Reads the same publishable config the browser uses, straight from
web/.env.local, so this needs no secret of its own.
"""

import io
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = Path(os.environ.get("WEB_ENV_FILE", HERE.parent / "web" / ".env.local"))
OUT = Path(os.environ.get("PASSAGES_FILE", HERE / "passages.txt"))


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def anonymous_token(url: str, key: str) -> str:
    request = urllib.request.Request(
        f"{url}/auth/v1/signup",
        data=b"{}",
        headers={"apikey": key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))["access_token"]


def fetch_passages(url: str, token: str, key: str) -> list[dict]:
    query = urllib.parse.urlencode(
        {"select": "level,difficulty,text_hi", "order": "level.asc,difficulty.asc"}
    )
    request = urllib.request.Request(
        f"{url}/rest/v1/passages?{query}",
        headers={"apikey": key, "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    if not ENV.is_file():
        print(f"No env file at {ENV}. Set WEB_ENV_FILE to point at it.")
        return 1

    env = read_env(ENV)
    try:
        url = env["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/")
        key = env["NEXT_PUBLIC_SUPABASE_ANON_KEY"]
    except KeyError as missing:
        print(f"{ENV} has no {missing}")
        return 1

    rows = fetch_passages(url, anonymous_token(url, key), key)
    if not rows:
        print("The passages table read back empty. Seed it (supabase/seed.sql) first.")
        return 1

    lines = []
    for row in rows:
        text = " ".join((row.get("text_hi") or "").split())
        if not text:
            continue
        lines.append(text)
        print(
            f"  [{row.get('level')}] difficulty {row.get('difficulty')}  "
            f"{len(text.split()):>3} words  {text[:40]}…"
        )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\n{len(lines)} passages -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
