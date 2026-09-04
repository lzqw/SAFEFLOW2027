#!/usr/bin/env bash
# Capacity-gated, checksum-verified transfer of final SHAPO P4.7 evidence to the host.

set -euo pipefail

REMOTE_HOST=${1:-seeta-nmb2}
REMOTE_RUN_ROOT=${2:-/root/autodl-tmp/SAFE/shapo_p47_four_env/formal}
LOCAL_OUTPUT_ROOT=${3:-}
RESERVE_BYTES=${4:-536870912}

fail() {
    printf '%s\n' "$1" >&2
    exit 2
}

[[ "$REMOTE_HOST" =~ ^[A-Za-z0-9_.@-]+$ ]] || fail "Unsafe SSH host or alias"
[[ "$REMOTE_RUN_ROOT" == /* ]] || fail "Remote run root must be absolute"
[[ -n "$LOCAL_OUTPUT_ROOT" ]] || \
    fail "Usage: $0 [ssh-alias] [remote-run-root] LOCAL_OUTPUT_ROOT [reserve-bytes]"
[[ "$RESERVE_BYTES" =~ ^[0-9]+$ ]] || fail "Reserve bytes must be a non-negative integer"
command -v ssh >/dev/null 2>&1 || fail "ssh is unavailable on the host"
command -v rsync >/dev/null 2>&1 || fail "rsync is unavailable on the host"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is unavailable on the host"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST")
read_remote_metadata() {
    "${SSH[@]}" bash -s -- "$REMOTE_RUN_ROOT" <<'REMOTE'
set -euo pipefail
run_root=$1
public_root=$run_root/public_artifacts
[[ -d "$public_root" ]] || {
    printf 'Final public artifact directory is absent: %s\n' "$public_root" >&2
    exit 2
}
[[ -f "$public_root/finalization_complete" ]] || {
    printf 'Final public artifact completion seal is absent: %s\n' "$public_root" >&2
    exit 2
}
[[ -f "$public_root/checksums.sha256" ]] || {
    printf 'Final public artifact checksum manifest is absent: %s\n' "$public_root" >&2
    exit 2
}
(cd "$public_root" && sha256sum -c checksums.sha256 >/dev/null)

mapfile -t watchdog_dirs < <(
    find "$run_root" -maxdepth 1 -mindepth 1 -type d -name 'watchdog-*' -print | sort
)
[[ ${#watchdog_dirs[@]} -eq 1 ]] || {
    printf 'Expected one watchdog directory below %s, found %s\n' \
        "$run_root" "${#watchdog_dirs[@]}" >&2
    exit 2
}
watchdog_root=${watchdog_dirs[0]}
health_files=(
    completion_health.json
    latest_health.json
    health_journal.jsonl
    launch_audit.json
    persistence_audit.json
)
for name in "${health_files[@]}"; do
    path=$watchdog_root/$name
    [[ -f "$path" ]] || {
        printf 'Required completion health artifact is absent: %s\n' "$path" >&2
        exit 2
    }
done

printf 'public_path\t%s\n' "$public_root"
printf 'public_bytes\t%s\n' "$(du -sb "$public_root" | cut -f1)"
printf 'watchdog_path\t%s\n' "$watchdog_root"
for name in "${health_files[@]}"; do
    path=$watchdog_root/$name
    printf 'health\t%s\t%s\t%s\n' \
        "$name" "$(sha256sum "$path" | cut -d' ' -f1)" "$(stat -c '%s' "$path")"
done
REMOTE
}

if ! REMOTE_METADATA_TEXT=$(read_remote_metadata); then
    fail "Remote final evidence is unavailable or failed source verification"
fi
mapfile -t REMOTE_METADATA <<<"$REMOTE_METADATA_TEXT"
REMOTE_PUBLIC_PATH=
REMOTE_PUBLIC_BYTES=
REMOTE_WATCHDOG_PATH=
declare -A HEALTH_SHA256=()
declare -A HEALTH_BYTES=()
for line in "${REMOTE_METADATA[@]}"; do
    IFS=$'\t' read -r kind first second third <<<"$line"
    case "$kind" in
        public_path) REMOTE_PUBLIC_PATH=$first ;;
        public_bytes) REMOTE_PUBLIC_BYTES=$first ;;
        watchdog_path) REMOTE_WATCHDOG_PATH=$first ;;
        health)
            HEALTH_SHA256[$first]=$second
            HEALTH_BYTES[$first]=$third
            ;;
        *) fail "Unexpected remote metadata record: $kind" ;;
    esac
done
[[ "$REMOTE_PUBLIC_PATH" == "$REMOTE_RUN_ROOT/public_artifacts" ]] || \
    fail "Remote public artifact path changed during discovery"
[[ "$REMOTE_WATCHDOG_PATH" == "$REMOTE_RUN_ROOT"/watchdog-* ]] || \
    fail "Remote watchdog path is outside the formal run root"
[[ "$REMOTE_PUBLIC_BYTES" =~ ^[0-9]+$ ]] || fail "Remote public artifact size is invalid"

HEALTH_NAMES=(
    completion_health.json
    latest_health.json
    health_journal.jsonl
    launch_audit.json
    persistence_audit.json
)
HEALTH_TOTAL_BYTES=0
for name in "${HEALTH_NAMES[@]}"; do
    [[ "${HEALTH_SHA256[$name]:-}" =~ ^[0-9a-f]{64}$ ]] || \
        fail "Remote health digest is invalid for $name"
    [[ "${HEALTH_BYTES[$name]:-}" =~ ^[0-9]+$ ]] || \
        fail "Remote health size is invalid for $name"
    HEALTH_TOTAL_BYTES=$((HEALTH_TOTAL_BYTES + HEALTH_BYTES[$name]))
done

LOCAL_OUTPUT_NAME=$(basename "$LOCAL_OUTPUT_ROOT")
[[ "$LOCAL_OUTPUT_NAME" != . && "$LOCAL_OUTPUT_NAME" != .. ]] || \
    fail "Local output path must identify a new directory"
LOCAL_OUTPUT_PARENT=$(dirname "$LOCAL_OUTPUT_ROOT")
mkdir -p "$LOCAL_OUTPUT_PARENT"
LOCAL_OUTPUT_PARENT=$(realpath -e "$LOCAL_OUTPUT_PARENT")
LOCAL_OUTPUT_ROOT=$LOCAL_OUTPUT_PARENT/$LOCAL_OUTPUT_NAME
[[ ! -e "$LOCAL_OUTPUT_ROOT" ]] || fail "Refusing to overwrite host artifacts: $LOCAL_OUTPUT_ROOT"
AVAILABLE_BYTES=$(df -PB1 "$LOCAL_OUTPUT_PARENT" | awk 'NR == 2 {print $4}')
[[ "$AVAILABLE_BYTES" =~ ^[0-9]+$ ]] || fail "Unable to determine host free space"
TRANSFER_BYTES=$((REMOTE_PUBLIC_BYTES + HEALTH_TOTAL_BYTES))
REQUIRED_BYTES=$((TRANSFER_BYTES + RESERVE_BYTES))
if (( REQUIRED_BYTES > AVAILABLE_BYTES )); then
    fail "Insufficient host space: need $REQUIRED_BYTES bytes including reserve, have $AVAILABLE_BYTES"
fi

STAGING_ROOT=$(mktemp -d "$LOCAL_OUTPUT_PARENT/.shapo-p47-sync.XXXXXX")
SYNC_SUCCEEDED=0
report_staging_on_failure() {
    local status=$?
    if [[ "$SYNC_SUCCEEDED" -ne 1 ]]; then
        printf 'SHAPO P4.7 host sync failed; staging retained at %s\n' "$STAGING_ROOT" >&2
    fi
    return "$status"
}
trap report_staging_on_failure EXIT

mkdir -p "$STAGING_ROOT/public_artifacts" "$STAGING_ROOT/runtime_health"
RSYNC_SSH='ssh -o BatchMode=yes -o ConnectTimeout=10'
rsync -a --protect-args -e "$RSYNC_SSH" -- \
    "$REMOTE_HOST:$REMOTE_PUBLIC_PATH/" "$STAGING_ROOT/public_artifacts/"
(
    cd "$STAGING_ROOT/public_artifacts"
    sha256sum -c checksums.sha256 >/dev/null
)

HEALTH_MANIFEST=$STAGING_ROOT/runtime_health_manifest.csv
printf '%s\n' 'artifact,remote_path,sha256,bytes' >"$HEALTH_MANIFEST"
for name in "${HEALTH_NAMES[@]}"; do
    local_path=$STAGING_ROOT/runtime_health/$name
    remote_path=$REMOTE_WATCHDOG_PATH/$name
    rsync -a --protect-args -e "$RSYNC_SSH" -- \
        "$REMOTE_HOST:$remote_path" "$local_path"
    local_sha256=$(sha256sum "$local_path" | cut -d' ' -f1)
    local_bytes=$(stat -c '%s' "$local_path")
    [[ "$local_sha256" == "${HEALTH_SHA256[$name]}" ]] || \
        fail "Host health digest mismatch for $name"
    [[ "$local_bytes" == "${HEALTH_BYTES[$name]}" ]] || \
        fail "Host health size mismatch for $name"
    printf '%s,%s,%s,%s\n' \
        "$name" "$remote_path" "$local_sha256" "$local_bytes" >>"$HEALTH_MANIFEST"
done

PUBLIC_MANIFEST_SHA256=$(sha256sum "$STAGING_ROOT/public_artifacts/checksums.sha256" | cut -d' ' -f1)
TRANSFER_MANIFEST=$STAGING_ROOT/transfer_manifest.tsv
{
    printf 'remote_host\t%s\n' "$REMOTE_HOST"
    printf 'remote_run_root\t%s\n' "$REMOTE_RUN_ROOT"
    printf 'remote_public_bytes\t%s\n' "$REMOTE_PUBLIC_BYTES"
    printf 'runtime_health_bytes\t%s\n' "$HEALTH_TOTAL_BYTES"
    printf 'transfer_bytes\t%s\n' "$TRANSFER_BYTES"
    printf 'host_available_bytes_before_transfer\t%s\n' "$AVAILABLE_BYTES"
    printf 'required_reserve_bytes\t%s\n' "$RESERVE_BYTES"
    printf 'public_checksum_manifest_sha256\t%s\n' "$PUBLIC_MANIFEST_SHA256"
} >"$TRANSFER_MANIFEST"

(
    cd "$STAGING_ROOT"
    find . -type f ! -name transfer_checksums.sha256 ! -name transfer_complete -print0 \
        | sort -z \
        | xargs -0 sha256sum >transfer_checksums.sha256
    sha256sum -c transfer_checksums.sha256 >/dev/null
)
touch "$STAGING_ROOT/transfer_complete"
mv -- "$STAGING_ROOT" "$LOCAL_OUTPUT_ROOT"
SYNC_SUCCEEDED=1
printf 'SHAPO P4.7 final evidence synchronized: %s (%s bytes)\n' \
    "$LOCAL_OUTPUT_ROOT" "$TRANSFER_BYTES"
