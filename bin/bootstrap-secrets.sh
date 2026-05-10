#!/bin/bash
# SOPS-decryption hook — stubbed in M1.
#
# Aegir M1 has no external credentials (leaderboard is read-only, air-gap
# assumption). When M2 introduces S3-backed model checkpoint mirrors,
# Anthropic API access, or Cloudera IDBroker tokens, this script will
# decrypt ``.env.cai.enc`` into ``.env.cai`` using SOPS + the platform's
# KMS binding. For now, a no-op.
#
# See ``~/local/src/zndx/atelier/bin/bootstrap-secrets.sh`` for the full
# pattern when you're ready to wire it up.

set -euo pipefail

if [ ! -f .env.cai.enc ]; then
  # Silent no-op when there's nothing to decrypt.
  exit 0
fi

if ! command -v sops >/dev/null 2>&1; then
  echo "WARNING: .env.cai.enc exists but sops is not on PATH; skipping decryption" >&2
  exit 0
fi

echo "Decrypting .env.cai.enc -> .env.cai"
sops --decrypt .env.cai.enc > .env.cai
chmod 0600 .env.cai
