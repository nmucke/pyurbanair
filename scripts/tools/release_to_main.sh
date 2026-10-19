#!/usr/bin/env bash
# Release dev into main: merge origin/dev into origin/main, delete the dev-only
# paths listed in .devonly, and push the result as a release/<date> branch to
# open a pull request into main from.
#
#   bash scripts/tools/release_to_main.sh
#
# Works in a throwaway worktree under .temp/, so your checkout is untouched.
# Conflicts on dev-only paths (edited on dev, deleted on main) resolve by
# deleting them; any other conflict stops the script with the worktree kept.
# Merge the pull request with "Create a merge commit", never squash or rebase:
# main must keep dev as a parent, or the next release conflicts on everything.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
git fetch -q origin main dev
branch=release/$(date +%Y-%m-%d)
mkdir -p .temp
worktree=$(mktemp -d "$PWD/.temp/release.XXXXXX")
git worktree add -q -b "$branch" "$worktree" origin/main
cd "$worktree"

git merge -q --no-ff --no-commit origin/dev || true
if ! git rev-parse -q --verify MERGE_HEAD >/dev/null; then
    echo "main already contains dev, or the merge failed; see $worktree" >&2
    exit 1
fi
git show origin/dev:.devonly | grep -v -e '^#' -e '^$' | while read -r path; do
    git rm -r -q -f --ignore-unmatch -- "$path"
done

conflicts=$(git diff --name-only --diff-filter=U)
if [[ -n $conflicts ]]; then
    echo "Resolve these conflicts in $worktree, then commit and push $branch:" >&2
    echo "$conflicts" >&2
    exit 1
fi
# --no-verify: dev's changes already passed the hooks on their own PRs.
git commit -q --no-verify -m "Release dev ($(git rev-parse --short origin/dev)) into main"
git push -q -u origin "$branch"

cd - >/dev/null
git worktree remove --force "$worktree"
url=$(git remote get-url origin)
url=${url%.git}
url=${url/git@github.com:/https://github.com/}
echo "Open the pull request: $url/compare/main...$branch?expand=1"
