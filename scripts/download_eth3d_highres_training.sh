#!/usr/bin/env bash
set -euo pipefail

ROOT="data/raw/eth3d"
KEEP_ARCHIVES=0
FORCE=0
BASE_URL="https://www.eth3d.net/data"

SCENES=(
  courtyard
  delivery_area
  electro
  facade
  kicker
  meadow
  office
  pipes
  playground
  relief
  relief_2
  terrace
  terrains
)

usage() {
  cat <<'EOF'
Download ETH3D high-res multi-view training DSLR JPG images and rendered depth maps.

Usage:
  bash scripts/download_eth3d_highres_training.sh [options]

Options:
  --root DIR          Output root. Default: data/raw/eth3d
  --scenes LIST       Comma-separated scene list. Default: all training scenes
  --keep-archives     Keep .7z archives after successful extraction
  --force             Re-download and re-extract even if scene folders exist
  -h, --help          Show this help

Examples:
  bash scripts/download_eth3d_highres_training.sh
  bash scripts/download_eth3d_highres_training.sh --root /openbayes/input/input0/eth3d
  bash scripts/download_eth3d_highres_training.sh --scenes pipes,meadow,terrace
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download() {
  local url="$1"
  local archive="$2"

  if have_cmd wget; then
    wget -c --tries=10 --timeout=60 "$url" -O "$archive"
  elif have_cmd curl; then
    curl -L --retry 10 --retry-delay 5 -C - "$url" -o "$archive"
  else
    die "Need wget or curl for downloads."
  fi
}

extract_archive() {
  local archive="$1"
  7z x -y "$archive"
}

scene_done() {
  local scene="$1"
  [[ -d "$scene/dslr_images" && -d "$scene/dslr_depth" ]]
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --root)
        ROOT="$2"
        shift 2
        ;;
      --scenes)
        IFS=',' read -r -a SCENES <<< "$2"
        shift 2
        ;;
      --keep-archives)
        KEEP_ARCHIVES=1
        shift
        ;;
      --force)
        FORCE=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}

main() {
  parse_args "$@"

  have_cmd 7z || die "7z not found. Install it first: apt-get update && apt-get install -y p7zip-full"

  mkdir -p "$ROOT"
  cd "$ROOT"

  log "ETH3D root: $(pwd)"
  log "Scenes: ${SCENES[*]}"
  log "This downloads distorted DSLR JPGs plus rendered depth maps. Depth maps match dslr_images, not undistorted images."

  for scene in "${SCENES[@]}"; do
    [[ -n "$scene" ]] || continue
    if [[ "$FORCE" -eq 0 ]] && scene_done "$scene"; then
      log "Skip $scene: dslr_images and dslr_depth already exist."
      continue
    fi

    for kind in dslr_jpg dslr_depth; do
      archive="${scene}_${kind}.7z"
      url="${BASE_URL}/${archive}"
      log "Download $archive"
      download "$url" "$archive"

      log "Extract $archive"
      extract_archive "$archive"

      if [[ "$KEEP_ARCHIVES" -eq 0 ]]; then
        log "Remove $archive"
        rm -f "$archive"
      fi
    done

    scene_done "$scene" || die "Scene $scene did not produce expected $scene/dslr_images and $scene/dslr_depth folders."
    log "Done $scene"
  done

  log "All requested ETH3D scenes are ready."
  log "Next: python scripts/make_eth3d_filelist.py --root $ROOT --output data/annotations/eth3d_full.csv --image-list data/annotations/eth3d_full_images.txt"
}

main "$@"
