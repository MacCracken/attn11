#!/bin/sh
# Version bump for attn11 — single entry point for all version references.
# Mirrors the patra / cyrius version-bump pattern.
#
# Usage: ./scripts/version-bump.sh 0.2.0
#
# attn11 keeps VERSION as the single source of truth: cyrius.cyml carries
# `version = "${file:VERSION}"`, so the package version is *derived* from the
# VERSION file and never needs a separate edit (the CI docs gate resolves the
# template). This script therefore only touches VERSION and stubs CHANGELOG —
# all-or-nothing, so the human can't forget a site and trip the CI version
# consistency check (`VERSION != cyrius.cyml` / `version not in CHANGELOG`).

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Current: $(cat VERSION 2>/dev/null || echo '<no VERSION file>')"
    exit 1
fi

NEW="$1"
OLD=$(cat VERSION 2>/dev/null | tr -d '[:space:]' || echo '')

if [ -z "$OLD" ]; then
    echo "error: VERSION file missing or empty" >&2
    exit 1
fi

if [ "$NEW" = "$OLD" ]; then
    echo "Already at $OLD (no changes)"
    exit 0
fi

# Sanity: NEW looks like a semver
case "$NEW" in
    [0-9]*.[0-9]*.[0-9]*) ;;
    *) echo "error: '$NEW' does not look like a semver" >&2; exit 1 ;;
esac

# 1. VERSION file (source of truth; cyrius.cyml derives via ${file:VERSION})
echo "$NEW" > VERSION

# 2. CHANGELOG.md — add a dated stub if no entry for $NEW yet. Inserted right
#    after the "## [Unreleased]" line. The stub is intentionally empty so the
#    author writes the actual Added/Changed/Fixed sections — this script only
#    guarantees the version line appears (CI requires it).
if [ -f CHANGELOG.md ]; then
    if ! grep -q "## \[$NEW\]" CHANGELOG.md; then
        TODAY=$(date +%Y-%m-%d)
        awk -v new="$NEW" -v today="$TODAY" '
            /^## \[Unreleased\]/ && !inserted {
                print
                print ""
                print "## [" new "] - " today
                print ""
                print "**TODO:** describe this release."
                inserted = 1
                next
            }
            { print }
        ' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md
    fi
fi

echo "$OLD -> $NEW"
echo ""
echo "Updated:"
echo "  VERSION"
if grep -q "## \[$NEW\]" CHANGELOG.md 2>/dev/null; then
    echo "  CHANGELOG.md ([$NEW] stub)"
fi
echo ""
echo "Still manual:"
echo "  - CHANGELOG.md sections (Added/Changed/Fixed)"
echo "  - Bump the cyrius toolchain pin in cyrius.cyml if needed"
echo "    (\`cyrius = \"X.Y.Z\"\` — separate from package.version; only pin a"
echo "    RELEASED cyrius version or CI will 404 on the toolchain download)."
echo "  - Refresh docs/development/state.md."
echo ""
echo "Then: git tag $NEW && git push --tags  (release.yml verifies VERSION == tag)"
