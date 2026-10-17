#!/usr/bin/env bash
# Link checker for the syllabus site.
# Usage: ./scripts/link_check.sh [docs_dir]
# Extracts http(s) URLs from all Markdown files, checks each with curl,
# and prints a per-file report plus a summary.
set -uo pipefail

DOCS_DIR="${1:-docs}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# Extract unique URLs from Markdown inline links and raw text.
# Skips localhost and example.com placeholders.
find "$DOCS_DIR" -name '*.md' -print0 \
  | xargs -0 grep -ohE 'https?://[^)><" ]+' \
  | sed -E 's/[.,;:]+$//' \
  | grep -vE '^https?://(localhost|127\.0\.0\.1|example\.com)' \
  | sort -u > "$TMPDIR/urls.txt"

total=0; ok=0; bad=0
echo "Checking $(wc -l < "$TMPDIR/urls.txt") unique URLs..."
while IFS= read -r url; do
  total=$((total + 1))
  code="$(curl -sL -o /dev/null -w '%{http_code}' --max-time 25 -A 'Mozilla/5.0 (X11; Linux x86_64) link-check' "$url" || echo "000")"
  case "$code" in
    200|301|302|303|307|308|403|429)
      ok=$((ok + 1))
      ;;
    *)
      bad=$((bad + 1))
      echo "BROKEN [$code] $url"
      ;;
  esac
done < "$TMPDIR/urls.txt"

echo "----------------------------------------"
echo "Summary: $total checked, $ok ok (2xx/3xx or 403/429 paywall), $bad broken"
