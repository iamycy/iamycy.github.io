#!/usr/bin/env bash
# Merge the upstream Academic Pages template into the current branch, keeping our
# version of every personal path and ignoring upstream's sample content there.
set -euo pipefail
export LC_ALL=C

UPSTREAM=${UPSTREAM:-upstream/master}

# Paths that are ours: upstream edits/additions/deletions under these are discarded.
PERSONAL=(
  .gitignore
  _config.yml
  _data/navigation.yml
  _pages/about.md
  _pages/cv.md
  _includes/analytics-providers/custom.html
  _posts _publications _talks _teaching _portfolio files
)

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Commit or stash your changes first." >&2
  exit 1
fi

git fetch upstream
git merge --no-ff --no-commit "$UPSTREAM" || true

# Reset personal paths to our side: drop files upstream added there, clear
# conflicted index entries, then restore our committed versions.
comm -23 <(git ls-files -- "${PERSONAL[@]}" | sort -u) \
         <(git ls-tree -r --name-only HEAD -- "${PERSONAL[@]}" | sort -u) |
  while IFS= read -r f; do rm -f -- "$f"; done
git rm -r -q --cached --ignore-unmatch -- "${PERSONAL[@]}"
git checkout HEAD -- "${PERSONAL[@]}"

conflicts=$(git diff --name-only --diff-filter=U)
if [ -n "$conflicts" ]; then
  echo "Unresolved conflicts outside personal paths:" >&2
  echo "$conflicts" >&2
  exit 1
fi

git commit --no-edit
echo "Merged $UPSTREAM. Preview with: bundle exec jekyll serve"
