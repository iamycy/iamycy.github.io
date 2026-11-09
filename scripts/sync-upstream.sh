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
  _sass/_custom.scss
  _posts _publications _talks _teaching _portfolio files
)

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Commit or stash your changes first." >&2
  exit 1
fi

# Start from the latest pushed state so the sync doesn't diverge from origin.
if ! git pull --no-rebase --no-edit; then
  echo "git pull failed; resolve it, then rerun this script." >&2
  exit 1
fi

git fetch upstream
git merge --no-ff --no-commit "$UPSTREAM" || true

if ! git rev-parse -q --verify MERGE_HEAD >/dev/null; then
  echo "Already up to date with $UPSTREAM."
  exit 0
fi

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

# Theme behaviour changes (JS, styles, layouts) merge silently without conflicts,
# so show them for review before committing.
THEME=(assets/js assets/css _sass _layouts _includes package.json Gemfile)
total=$(git rev-list --count --no-merges HEAD..MERGE_HEAD)
echo "Merging $total upstream commits. Those touching theme files:"
git --no-pager log --oneline --no-merges HEAD..MERGE_HEAD -- "${THEME[@]}"
echo
echo "Theme changes staged by this merge:"
git --no-pager diff --cached --stat HEAD -- "${THEME[@]}"
echo

if [ ! -t 0 ]; then
  echo "Not interactive: merge left uncommitted. Review it, then run" >&2
  echo "'git commit --no-edit' or 'git merge --abort'." >&2
  exit 1
fi
read -r -p "Commit this merge? [y/N] " answer
if [ "$answer" != y ] && [ "$answer" != Y ]; then
  git merge --abort
  echo "Merge aborted."
  exit 1
fi

git commit --no-edit
echo "Merged $UPSTREAM. Preview with: bundle exec jekyll serve"
