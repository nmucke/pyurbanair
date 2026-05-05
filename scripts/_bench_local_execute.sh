#!/usr/bin/env bash
# Benchmark-instrumented variant of libs/pyudales/shell_scripts/local_execute.sh.
# Writes per-stage timings (copy / mpiexec / gather_outputs) to
# $BENCH_TIMING_DIR/${exp}.timings, one "<stage> <seconds>" per line.

set -e
set -o pipefail

if (( $# < 1 )); then
    echo "The path to case folder must be set."
    exit 1
fi

pushd "$1" > /dev/null
inputdir=$(pwd)
exp="${inputdir: -3}"

if [ -f config.sh ]; then
    source config.sh
fi
if [ -z "$NCPU" ]; then NCPU=1; fi
if [ -z "$DA_WORKDIR" ]; then echo "DA_WORKDIR unset"; exit 1; fi
if [ -z "$DA_BUILD" ]; then echo "DA_BUILD unset"; exit 1; fi
if [ -z "$DA_TOOLSDIR" ]; then echo "DA_TOOLSDIR unset"; exit 1; fi

outdir=$DA_WORKDIR/$exp

BENCH_DIR="${BENCH_TIMING_DIR:-/tmp/bench_timings}"
mkdir -p "$BENCH_DIR"
TLOG="$BENCH_DIR/${exp}.timings"
: > "$TLOG"

now() { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f", b - a }'; }

mkdir -p "$outdir"

t0=$(now)
cp -r ./* "$outdir"
t1=$(now)
echo "copy $(elapsed "$t0" "$t1")" >> "$TLOG"

pushd "$outdir" > /dev/null

t2=$(now)
set +e
mpiexec -n "$NCPU" --bind-to none --oversubscribe "$DA_BUILD" "namoptions.$exp" \
    >> "run.$exp.log" 2>&1
mpi_exit=$?
set -e
t3=$(now)
echo "mpiexec $(elapsed "$t2" "$t3")" >> "$TLOG"

# DelftBlue: conda-forge OpenMPI can SIGSEGV inside MPI_Finalize even after
# a clean uDALES run (UCX/InfiniBand tear-down). Tolerate exit 139 if
# uDALES printed its "TOTAL CPU time" end-of-run line. Real crashes still
# propagate.
if [ "$mpi_exit" -ne 0 ]; then
    if [ "$mpi_exit" -eq 139 ] && grep -q "TOTAL CPU time" "run.$exp.log"; then
        echo "mpiexec exit=139 after clean uDALES run; tolerating finalize segfault." >&2
    else
        echo "mpiexec failed with exit $mpi_exit" >&2
        exit "$mpi_exit"
    fi
fi

t4=$(now)
"$DA_TOOLSDIR/gather_outputs.sh" "$outdir" > gather.log 2>&1
t5=$(now)
echo "gather $(elapsed "$t4" "$t5")" >> "$TLOG"

popd > /dev/null
popd > /dev/null

echo "total $(elapsed "$t0" "$t5")" >> "$TLOG"
