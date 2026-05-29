#!/usr/bin/env python3
"""Generate a Telethon StringSession for non-interactive auth.

Run this ONCE locally. It logs into your Telegram *user* account (the one that
can read the source channel) and prints a session string. Put that value in
.env as TELEGRAM_SESSION_STRING so the container never needs interactive login.

Usage:
    export TELEGRAM_API_ID=...
    export TELEGRAM_API_HASH=...
    python scripts/generate_session.py
"""

from __future__ import annotations

import os
import sys

from telethon.sync import TelegramClient
from telethon.sessions import StringSession


def main() -> int:
    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        print("Set TELEGRAM_API_ID and TELEGRAM_API_HASH first.", file=sys.stderr)
        return 1

    print("Logging in. You'll be asked for your phone number and the code Telegram sends you.\n")
    with TelegramClient(StringSession(), int(api_id), api_hash) as client:
        session_string = client.session.save()
        me = client.get_me()
        print("\n" + "=" * 60)
        print(f"Logged in as: {me.first_name} (@{me.username})")
        print("=" * 60)
        print("\nTELEGRAM_SESSION_STRING=" + session_string)
        print("\nStore this in your .env (and keep it secret).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
