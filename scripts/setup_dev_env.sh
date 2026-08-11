#!/usr/bin/env bash
# Bootstrap the pixi `dev` environment, working around the bin/test clobber.
#
# `tempest-remap` (pulled in by `nco` in the udales feature) ships scripts under
# `bin/test/` as a DIRECTORY, while `coreutils` ships a `bin/test` FILE. Only one
# of them can own the path, so `pixi install -e dev` aborts whenever it has to
# create or remove it. That happens in two different situations, which need two
# different repairs:
#
#   1. First install into an empty env. coreutils lands first, then tempest-remap
#      cannot create its directory. Deleting the file and re-running lets
#      tempest-remap claim the path -- this is the long-standing workaround.
#   2. Re-install over an existing env (the CI cache, or any env whose solve has
#      moved on). pixi wants to remove the old coreutils, whose recorded file
#      list still claims `bin/test`, but a tempest-remap DIRECTORY sits there
#      now. `rm -f` cannot remove a directory, so repair 1 fails outright.
#
# Note that a successful bootstrap always leaves the env in state 2: coreutils'
# conda-meta record claims a file that is really tempest-remap's directory. The
# env works, but the next install that touches coreutils trips over it -- which
# is why this script ends with a full rebuild rather than giving up.
#
# Escalates: plain install -> drop whatever occupies bin/test -> rebuild the env
# from scratch. Safe to re-run; no-ops once the env is good.
set -uo pipefail
cd "$(dirname "$0")/.."

ENV_DIR=".pixi/envs/dev"

try_install () {
    pixi install -e dev
}

if try_install; then
    exit 0
fi

echo "setup-dev: install failed; clearing ${ENV_DIR}/bin/test and retrying." >&2
rm -rf "${ENV_DIR}/bin/test"
if try_install; then
    exit 0
fi

# The env is inconsistent in a way an in-place repair cannot reach (a stale
# coreutils record, a half-applied upgrade). Rebuilding costs a full download,
# but it is the only state the two-step workaround is known to converge from.
echo "setup-dev: still failing; rebuilding ${ENV_DIR} from scratch." >&2
rm -rf "${ENV_DIR}"
try_install || {
    rm -rf "${ENV_DIR}/bin/test"
    try_install
}
