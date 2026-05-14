#!/usr/bin/env bash
# release.sh — cut a measured release of model-benchbox.
#
# Runs the full two-step release pipeline against tinfoilsh/model-benchbox:
#   1. tinfoil-build.yml — builds the Docker image, pushes to ghcr.io, runs
#      update-container-action to rewrite tinfoil-config.yml's digest and
#      create the version tag.
#   2. tinfoil-release.yml — chained automatically by the build workflow;
#      runs measure-image-action and registers the release in the Sigstore
#      transparency log.
#
# Usage:
#   ./release.sh                 auto-bump patch from highest existing v*.*.* tag
#   ./release.sh v0.0.5          explicit version
#   ./release.sh v0.0.5 --force  delete a pre-existing tag without prompting
#
# Requires: gh + git, with a gh session authorized for tinfoilsh/model-benchbox.

set -euo pipefail

REPO="atlaie/cc-benchbox"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VERSION=""
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        v*) VERSION="$arg" ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

step() { printf '\n==> %s\n' "$*"; }
fail() { printf '[fail] %s\n' "$*" >&2; exit 1; }

step "Pre-flight"
gh auth status &>/dev/null || fail "gh not authenticated. Run: gh auth login"
git fetch origin --tags --quiet

if [ -z "$VERSION" ]; then
    LATEST="$(git tag -l 'v[0-9]*.[0-9]*.[0-9]*' | sort -V | tail -1)"
    [ -n "$LATEST" ] || fail "No existing v*.*.* tag — pass one explicitly: ./release.sh v0.0.1"
    IFS='.' read -r MAJ MIN PAT <<<"${LATEST#v}"
    VERSION="v${MAJ}.${MIN}.$((PAT + 1))"
    echo "Auto-bumped from $LATEST -> $VERSION"
fi

if git ls-remote --tags origin "refs/tags/$VERSION" | grep -q .; then
    if [ "$FORCE" -eq 0 ]; then
        printf 'Tag %s exists on origin. Delete and retry? [y/N] ' "$VERSION"
        read -r ans
        case "$ans" in y|Y|yes) ;; *) fail "Aborted." ;; esac
    fi
    git tag -d "$VERSION" 2>/dev/null || true
    git push origin ":refs/tags/$VERSION"
    echo "Deleted $VERSION from origin"
fi

step "Triggering tinfoil-build.yml ($VERSION)"
PREV_RELEASE="$(gh run list -R "$REPO" --workflow=tinfoil-release.yml --limit 1 --json databaseId --jq '.[0].databaseId // ""')"
gh workflow run tinfoil-build.yml -R "$REPO" -f version="$VERSION"

# Wait for the dispatched build run to appear (gh queues it asynchronously)
BUILD_RUN=""
for _ in {1..15}; do
    sleep 2
    BUILD_RUN="$(gh run list -R "$REPO" --workflow=tinfoil-build.yml --limit 1 --json databaseId --jq '.[0].databaseId // ""')"
    [ -n "$BUILD_RUN" ] && break
done
[ -n "$BUILD_RUN" ] || fail "Build run did not appear within 30s"

step "Watching build run https://github.com/$REPO/actions/runs/$BUILD_RUN"
gh run watch -R "$REPO" "$BUILD_RUN" --exit-status \
    || fail "tinfoil-build.yml failed. See https://github.com/$REPO/actions/runs/$BUILD_RUN"

step "Waiting for tinfoil-release.yml to start"
RELEASE_RUN=""
for _ in {1..60}; do
    NEW="$(gh run list -R "$REPO" --workflow=tinfoil-release.yml --limit 1 --json databaseId --jq '.[0].databaseId // ""')"
    if [ -n "$NEW" ] && [ "$NEW" != "$PREV_RELEASE" ]; then
        RELEASE_RUN="$NEW"
        break
    fi
    sleep 3
done
[ -n "$RELEASE_RUN" ] || fail "Release run did not start. Manually trigger: gh workflow run tinfoil-release.yml -R $REPO --ref $VERSION"

step "Watching release run https://github.com/$REPO/actions/runs/$RELEASE_RUN"
gh run watch -R "$REPO" "$RELEASE_RUN" --exit-status \
    || fail "tinfoil-release.yml failed. See https://github.com/$REPO/actions/runs/$RELEASE_RUN"

step "Released $VERSION"
cat <<EOF

  Image:  ghcr.io/$REPO:$VERSION
  Tag:    https://github.com/$REPO/releases/tag/$VERSION

Deploy fresh:
  tinfoil container create benchbox \\
      --repo $REPO --tag $VERSION --debug \\
      --ssh-key laptop \\
      --secret GITHUB_TOKEN --secret HF_TOKEN --secret TINFOIL_API_KEY

Update existing:
  tinfoil container relaunch benchbox --tag $VERSION
EOF
