#!/usr/bin/env bash
# Fetch the Children's Song Dataset (Choi et al., ISMIR 2020; Zenodo 10.5281/zenodo.4916302,
# v1.1) and unpack its English half, for `chimera compare csd`.
# License CC BY-NC-SA 4.0: research use; don't ship its audio or anything derived from it.
#   tools/fetch-csd.sh [dir]     (default ~/datasets/csd; the songs land in <dir>/CSD/english)
set -euo pipefail
dir="${1:-$HOME/datasets/csd}"
url="https://zenodo.org/api/records/4916302/files/CSD.zip/content"
md5="74d121dd8706fded26a15526a379f7a2"
mkdir -p "$dir"
cd "$dir"
if ! echo "$md5  CSD.zip" | md5sum -c --status 2>/dev/null; then
  echo "downloading CSD.zip (1.85 GB) to $dir"
  curl -fL -C - -o CSD.zip "$url"
  echo "$md5  CSD.zip" | md5sum -c
fi
unzip -qo CSD.zip 'CSD/english/*'
echo "unpacked $(ls CSD/english/csv | wc -l) English annotations to $dir/CSD/english"
