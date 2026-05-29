# NOBS

A small, dockerized Python service that runs on a GCP `e2-micro` VM and does two things:

1. **Daily news digest** — fetches RSS/Atom feeds from N sources, runs each
   through **Gemini Flash** with a per-source prompt, translates the result to
   Ukrainian, and posts a formatted summary to a Telegram channel.
2. **Channel monitor** — uses a Telegram **user account** (Telethon/MTProto) to
   watch a source channel, evaluates each new message with Gemini Flash
   (clickbait / low-value filter), and immediately forwards kept messages to the
   target channel via the bot.

Both run in a single asyncio process. Prompts, feeds and the schedule live in a
`config.yaml` on **GCS** and are re-read at runtime, so you can edit them without
a restart or redeploy. Secrets and fixed IDs live in environment variables.

## Architecture

```
app/
  main.py          # async entrypoint: APScheduler (digest) + Telethon (monitor)
  settings.py      # env vars (secrets, fixed IDs)
  config.py        # GCS config.yaml loader (TTL-cached)
  gemini.py        # Gemini Flash wrapper + exponential backoff (tenacity)
  digest.py        # Purpose 1: RSS -> Gemini -> Telegram
  monitor.py       # Purpose 2: Telethon -> Gemini -> forward
  telegram_bot.py  # python-telegram-bot publisher (chunking, retries)
  storage.py       # SQLite dedup (seen RSS entries + message ids)
scripts/
  generate_session.py  # make TELEGRAM_SESSION_STRING (run once, locally)
  provision_vm.sh      # create GCS bucket + e2-micro VM with Docker
  deploy.sh            # build image locally -> docker save|load to VM -> up -d
  upload_config.sh     # push config.yaml to GCS (live, no redeploy)
```

## Message format

```
{label}:
• {title} (link)
{brief summary}
```

(Rendered as Telegram HTML; the title links to the article.)

## Prerequisites

- A Telegram **bot** (via @BotFather) that is an **admin** of the destination channel.
- A Telegram **user account** that can read the source channel; API ID/HASH from
  <https://my.telegram.org/apps>.
- A **Gemini API key** (<https://aistudio.google.com/apikey>).
- A **GCP project** with billing, and the `gcloud` CLI authenticated locally.

## Setup

### 1. Configure environment

```bash
cp .env.example .env          # fill in tokens, ids, keys
cp scripts/deploy.env.example scripts/deploy.env   # project, zone, bucket
```

### 2. Generate the Telethon session string (once, locally)

```bash
pip install -r requirements.txt
export TELEGRAM_API_ID=... TELEGRAM_API_HASH=...
python scripts/generate_session.py
# paste the printed TELEGRAM_SESSION_STRING into .env
```

### 3. Provision GCP + upload config

```bash
./scripts/provision_vm.sh                 # bucket + e2-micro VM (Docker)
cp config.example.yaml config.yaml        # then edit feeds/prompts
./scripts/upload_config.sh config.yaml
```

**Free tier:** `scripts/deploy.env.example` defaults to the GCP *Always Free*
shape — one `e2-micro` in `us-central1` on a 10 GB `pd-standard` disk. To stay
free you must keep the zone in `us-west1` / `us-central1` / `us-east1`, the
machine type `e2-micro`, and the disk `pd-standard` (≤30 GB). `provision_vm.sh`
warns if any of these drift. The 1 GB instance gets a 2 GB swap file on first
boot so `docker build` doesn't OOM, and the bucket is created in the VM's region
to avoid cross-region egress. The VM region is independent of the digest
timezone (that's set in `config.yaml`).

### 4. Deploy

```bash
./scripts/deploy.sh
```

This **builds the image locally** for `linux/amd64`, ships it to the VM via
`docker save | ssh docker load` (no registry, no build on the 1 GB VM), copies
`docker-compose.yml` + `.env` (+ `secrets/` if present), and runs
`docker compose up -d` using the pre-loaded image.

> On Apple Silicon the `--platform linux/amd64` build is required so the image
> runs on the amd64 e2-micro. Override with `PLATFORM=...` or `IMAGE=...` env vars.

## Editing prompts / feeds without redeploy

Edit `config.yaml` and re-upload:

```bash
./scripts/upload_config.sh config.yaml
```

Changes take effect within `CONFIG_TTL_SECONDS` (default 60s) — no restart.

## Run locally

```bash
docker compose up --build
# or, bare metal:
pip install -r requirements.txt && python -m app.main
```

## Credentials on GCP

On the VM, leave `GOOGLE_APPLICATION_CREDENTIALS` empty in `.env` to use the VM's
attached service account (it needs `roles/storage.objectViewer` on the config
bucket — granted via `--scopes=storage-ro` + a suitable SA). Locally, point
`GOOGLE_APPLICATION_CREDENTIALS` at a key file mounted under `./secrets`.

## Environment variables

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token (posting/forwarding) |
| `TELEGRAM_CHANNEL_ID` | Destination channel (digests + forwards) |
| `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` | User account API creds (Telethon) |
| `TELEGRAM_PHONE` | User account phone (only for session generation) |
| `TELEGRAM_SESSION_STRING` | Non-interactive Telethon auth |
| `TELEGRAM_SOURCE_CHANNEL` | Channel the monitor watches (fixed) |
| `GEMINI_API_KEY` | Gemini Flash key |
| `GCS_BUCKET_NAME` / `GCS_CONFIG_PATH` | Where `config.yaml` lives |
| `CONFIG_TTL_SECONDS` | Config cache TTL (default 60) |
| `GOOGLE_APPLICATION_CREDENTIALS` | SA key path (empty on VM) |
| `DB_PATH` | SQLite dedup DB path |

## Notes & design decisions

- **Dedup**: SQLite tracks seen RSS entry ids and `chat_id:message_id` pairs so
  restarts don't repost/re-forward. Persisted in a Docker volume.
- **Backoff**: Gemini 429/5xx are retried with exponential backoff + jitter
  (up to 6 attempts) before bubbling up.
- **Failure alerts**: if the daily digest throws, an alert is posted to the channel.
- **Config TTL**: the monitor can be chatty, so config is cached briefly instead
  of fetching from GCS on literally every message. Set `CONFIG_TTL_SECONDS=0` to
  disable caching.
