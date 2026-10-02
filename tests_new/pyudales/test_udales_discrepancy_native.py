"""Resource integrity and the independently compiled native discrepancy kernel."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "libs/pyudales/src/pyudales/solver_extensions/discrepancy"
UPSTREAM = ROOT / "libs/pyudales/u-dales"


def test_discrepancy_resource_hashes() -> None:
    manifest = json.loads((RESOURCES / "manifest.json").read_text())
    assert manifest["upstream_commit"] == "b84916ac60cecd1da54dd09df76c15e30dcaabe9"
    assert manifest["capability"] == "sgs_strain_rotation_v1"
    for name, expected in manifest["resources"].items():
        assert hashlib.sha256((RESOURCES / name).read_bytes()).hexdigest() == expected


def test_discrepancy_patch_applies_to_pristine_source(tmp_path: Path) -> None:
    manifest = json.loads((RESOURCES / "manifest.json").read_text())
    originals = {}
    for name, expected in manifest["inputs"].items():
        result = subprocess.run(
            [
                "git",
                "-C",
                str(UPSTREAM),
                "show",
                f"{manifest['upstream_commit']}:{name}",
            ],
            capture_output=True,
        )
        if result.returncode:
            pytest.skip("Pinned uDALES source is not available locally")
        originals[name] = result.stdout.decode()
        assert hashlib.sha256(result.stdout).hexdigest() == expected
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(result.stdout)
    subprocess.run(
        ["git", "apply", "--check", str(RESOURCES / manifest["patch"])],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "apply", str(RESOURCES / manifest["patch"])], cwd=tmp_path, check=True
    )
    for resource, destination in manifest["copies"].items():
        shutil.copyfile(RESOURCES / resource, tmp_path / destination)
    for name, expected in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected

    subgrid = (tmp_path / "src/modsubgrid.f90").read_text()
    # The actual upstream momentum stress-divergence routines are untouched.
    assert (
        subgrid[subgrid.index("  subroutine sources") :]
        == originals["src/modsubgrid.f90"][
            originals["src/modsubgrid.f90"].index("  subroutine sources") :
        ]
    )
    vreman = subgrid[subgrid.index("    elseif(lvreman)") :]
    scalar = vreman.index("ekh(:,:,:) = ekm(:,:,:)*prandtli")
    correction = vreman.index("ekm(ib:ie,jb:je,kb:ke) =")
    molecular = vreman.index("ekm(:,:,:) = ekm(:,:,:) + numol")
    boundary = vreman.index("call closurebc")
    assert scalar < correction < molecular < boundary
    program = (tmp_path / "src/program.f90").read_text()
    assert program.index("if (lsgs_discrepancy) call halos") < program.index(
        "  call boundary"
    )
    assert program.index("    call closure") < program.index("    call tstep_update")
    assert program.index("call discrepancy_check_clocks") < program.index(
        "if (lsgs_discrepancy) call halos"
    )
    check = subgrid.split("  subroutine discrepancy_check_clocks", 1)[1].split(
        "  end subroutine discrepancy_check_clocks", 1
    )[0]
    assert "if (.not. lsgs_discrepancy) return" in check
    assert "MPI_ALLREDUCE(clocks,clocks_min" in check
    assert "MPI_ALLREDUCE(clocks,clocks_max" in check
    assert "any(clocks_min /= clocks_max)" in check
    assert "MPI_ABORT" in check


@pytest.mark.parametrize("default_real_8", [False, True])  # type: ignore[misc]
def test_native_discrepancy_kernel(tmp_path: Path, default_real_8: bool) -> None:
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is unavailable")
    assert compiler is not None
    driver = tmp_path / "kernel_test.f90"
    driver.write_text(
        """
program test_kernel
  use modsgsdiscrepancy
  use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan
  implicit none
  real :: a(3,3), q, phi, multiplier, saturation, expected, tol
  tol = 100.*epsilon(1.)
  sgs_discrepancy_height = 10.
  sgs_discrepancy_za_over_h = 0.5
  sgs_discrepancy_zb_over_h = 1.5
  sgs_discrepancy_epsilon = 0.1
  sgs_discrepancy_cap = log(3.)
  if (.not. discrepancy_valid()) stop 1
  a = 0.
  call discrepancy_features(a,10.,q,phi)
  if (q /= 0. .or. abs(phi-1.) > tol) stop 2
  call discrepancy_multiplier(a,10.,multiplier,saturation)
  if (multiplier /= 1. .or. saturation /= 0.) stop 3
  ! Pure strain: S:S=2, Omega:Omega=0.
  a(1,1) = 1.
  a(2,2) = -1.
  call discrepancy_features(a,10.,q,phi)
  if (abs(q+2./2.01) > tol) stop 4
  ! Solid-body rotation: Omega:Omega=2, S:S=0.
  a = 0.
  a(1,2) = -1.
  a(2,1) = 1.
  call discrepancy_features(a,10.,q,phi)
  if (abs(q-2./2.01) > tol) stop 5
  ! Simple shear and support endpoints.
  a(1,2) = 0.
  call discrepancy_features(a,5.,q,phi)
  if (q /= 0. .or. phi /= 0.) stop 6
  call discrepancy_features(a,15.,q,phi)
  if (phi /= 0.) stop 7
  call discrepancy_features(a,-1.,q,phi)
  if (phi /= 0.) stop 8
  sgs_bias_b0 = 0.2
  sgs_bias_b1 = 0.3
  sgs_bias_b2 = 0.4
  a = 0.
  a(1,1) = 1.
  a(2,2) = -1.
  expected = exp(log(3.)*tanh((0.2+0.3-0.4*2./2.01)/log(3.)))
  call discrepancy_multiplier(a,10.,multiplier,saturation)
  if (abs(multiplier-expected) > tol) stop 9
  ! Extreme finite coefficients saturate safely, including cancellation.
  sgs_bias_b0 = huge(1.)
  sgs_bias_b1 = 0.
  sgs_bias_b2 = 0.
  call discrepancy_multiplier(a,10.,multiplier,saturation)
  if (abs(multiplier-3.) > tol .or. saturation /= 1.) stop 10
  sgs_bias_b0 = -huge(1.)
  call discrepancy_multiplier(a,10.,multiplier,saturation)
  if (abs(multiplier-1./3.) > tol) stop 11
  sgs_bias_b1 = huge(1.)
  call discrepancy_multiplier(a,10.,multiplier,saturation)
  if (abs(multiplier-1.) > tol) stop 12
  sgs_discrepancy_height = 0.
  if (discrepancy_valid()) stop 13
  sgs_discrepancy_height = 10.
  sgs_discrepancy_epsilon = 0.
  if (discrepancy_valid()) stop 14
  sgs_discrepancy_epsilon = 0.1
  sgs_discrepancy_cap = log(huge(1.))
  if (discrepancy_valid()) stop 15
  sgs_discrepancy_cap = 1.
  sgs_bias_b0 = ieee_value(1.,ieee_quiet_nan)
  if (discrepancy_valid()) stop 16
end program test_kernel
"""
    )
    command = [compiler, "-std=f2008", "-ffree-line-length-none", "-fcheck=all"]
    if sys.platform == "darwin":
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        command.extend(["-isysroot", sdk])
    if default_real_8:
        command.append("-fdefault-real-8")
    # NaN validation itself must also work with upstream's FPE trap settings.
    command.extend(
        [
            "-ffpe-trap=invalid,zero,overflow",
            str(RESOURCES / "modsgsdiscrepancy.f90"),
            str(driver),
            "-o",
            str(tmp_path / "kernel_test"),
        ]
    )
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run([str(tmp_path / "kernel_test")], check=True, capture_output=True)


def test_native_vreman_zero_gradient_limit(tmp_path: Path) -> None:
    """Reproduce pristine 0/0 and run the exact enabled guard shipped in the patch."""
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is unavailable")
    assert compiler is not None
    additions = "\n".join(
        line[1:]
        for line in (RESOURCES / "discrepancy.patch").read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    start = additions.index("               if (aa == 0.) then")
    end = additions.index("               end if", start) + len("               end if")
    guard = additions[start:end]
    baseline = "ekm(i,j,k) = c_vreman*sqrt(max(bb/aa,0.))"
    for label, expression in [("stock", baseline), ("enabled", guard)]:
        source = tmp_path / f"{label}.f90"
        source.write_text(
            "program test_zero\nimplicit none\n"
            "real :: aa, bb, c_vreman, ekm(1,1,1)\ninteger :: i,j,k\n"
            "aa=0.\nbb=0.\nc_vreman=0.07\ni=1\nj=1\nk=1\n"
            + expression
            + "\nif (ekm(1,1,1) /= 0.) stop 1\nend program\n"
        )
        command = [compiler, "-fdefault-real-8", "-ffpe-trap=invalid,zero,overflow"]
        if sys.platform == "darwin":
            sdk = subprocess.check_output(
                ["xcrun", "--show-sdk-path"], text=True
            ).strip()
            command.extend(["-isysroot", sdk])
        executable = tmp_path / label
        command.extend([str(source), "-o", str(executable)])
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
        result = subprocess.run([str(executable)], cwd=tmp_path, capture_output=True)
        if label == "stock":
            assert (
                result.returncode != 0
            ), "Pristine Vreman must reproduce the zero/zero trap"
        else:
            assert result.returncode == 0, result.stderr.decode()
