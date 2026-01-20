"""Python implementation of MATLAB preprocessing class for uDALES."""

import os
import pathlib
import re
from typing import Any, Optional

import numpy as np


class Preprocessing:
    """Class for pre-processing in uDALES."""

    def __init__(self, expnr: int, exppath: Optional[str] = None) -> None:
        """
        Class constructor.

        Args:
            expnr: Integer equal to simulation number.
            exppath: Path to simulations directory.
        """
        self.cpath = os.getcwd()
        self.expnr = f"{expnr:03d}"

        if exppath is not None:
            dapath = os.path.join(exppath, self.expnr)
        else:
            dapath = self.expnr

        os.chdir(dapath)
        self.path = os.getcwd()

        namoptionsfile = os.path.join(self.path, f"namoptions.{self.expnr}")

        if not os.path.exists(namoptionsfile):
            raise FileNotFoundError(f"namoptions.{self.expnr} not found. Exiting...")

        # Read namoptions file
        self._read_namoptions(namoptionsfile)
        
        # Ensure iexpnr is set (it should be read from namoptions, but set default if not)
        if not hasattr(self, 'iexpnr'):
            self.iexpnr = expnr

        os.chdir(self.cpath)

    def _read_namoptions(self, filename: str) -> None:
        """Read namoptions file and set properties."""
        tokens = re.compile(r"(.*)\s*=\s*(.*)")
        white = re.compile(r"\s*")

        with open(filename, "r") as f:
            for line in f:
                match = tokens.match(line)
                if match:
                    lhs = white.sub("", match.group(1))
                    rhs = white.sub("", match.group(2)).rstrip(";").strip("[]'\"")
                    if rhs == ".false.":
                        setattr(self, lhs, False)
                    elif rhs == ".true.":
                        setattr(self, lhs, True)
                    elif rhs.isdigit() or (rhs.replace(".", "").replace("-", "").isdigit()):
                        try:
                            # Try to parse as number
                            if "." in rhs or "e" in rhs.lower() or "E" in rhs:
                                setattr(self, lhs, float(rhs))
                            else:
                                setattr(self, lhs, int(rhs))
                        except (ValueError, OverflowError):
                            # If parsing fails, keep as string
                            setattr(self, lhs, rhs)
                    else:
                        try:
                            val = float(rhs)
                            setattr(self, lhs, val)
                        except ValueError:
                            setattr(self, lhs, rhs)

    def addvar(self, lhs: str, var: Any) -> None:
        """Install a variable in the object."""
        if not hasattr(self, lhs):
            setattr(self, lhs, var)

    def gopath(self) -> None:
        """Go to simulation path."""
        os.chdir(self.path)

    def gohome(self) -> None:
        """Go to work path."""
        os.chdir(self.cpath)

    def chcpath(self, newpath: str) -> None:
        """Change work path."""
        here = os.getcwd()
        os.chdir(newpath)
        self.cpath = os.getcwd()
        os.chdir(here)

    @staticmethod
    def set_defaults(obj: "Preprocessing") -> None:
        """Set default values for preprocessing object."""
        # &RUN
        obj.addvar("ltrees", 0)
        obj.addvar("ltreesfile", 0)

        if obj.ltrees and not obj.ltreesfile:
            obj.addvar("tree_dz", 0)
            obj.addvar("tree_dx", 0)
            obj.addvar("tree_dy", 0)
            obj.addvar("tree_h", 0)
            obj.addvar("tree_w", 0)
            obj.addvar("tree_b", 0)
            obj.addvar("nrows", 0)

        if obj.ltrees and obj.ltreesfile:
            obj.addvar("treesfile", "")

        obj.addvar("lpurif", 0)
        if obj.lpurif:
            raise NotImplementedError("Purifiers not currently implemented")

        obj.addvar("luoutflowr", 0)
        obj.addvar("lvoutflowr", 0)
        obj.addvar("luvolflowr", 0)
        obj.addvar("lvvolflowr", 0)

        # &DOMAIN
        obj.addvar("itot", 64)
        obj.addvar("xlen", 64)
        obj.addvar("jtot", 64)
        obj.addvar("ylen", 64)
        obj.addvar("ktot", 96)

        obj.addvar("dx", obj.xlen / obj.itot)
        obj.addvar("dy", obj.ylen / obj.jtot)

        # BCs
        obj.addvar("BCxm", 1)
        obj.addvar("BCym", 1)

        # &ENERGYBALANCE
        obj.addvar("lEB", 0)
        obj.addvar("lfacTlyrs", 0)

        # &WALLS
        obj.addvar("iwallmom", 3)
        obj.addvar("iwalltemp", 1)
        obj.addvar("lbottom", 0)
        obj.addvar("lwritefac", 0)

        # &PHYSICS
        obj.addvar("ltempeq", 0)
        obj.addvar("lmoist", 0)
        obj.addvar("lchem", 0)
        obj.addvar("lprofforc", 0)
        obj.addvar("lcoriol", 0)
        obj.addvar("idriver", 0)

        if (
            not obj.luoutflowr
            and not obj.lvoutflowr
            and not obj.luvolflowr
            and not obj.lvvolflowr
            and not obj.lprofforc
            and not obj.lcoriol
            and obj.idriver != 2
        ):
            obj.addvar("ldp", 1)
            print("No forcing switch config. setup and not a driven simulation so initial velocities and/or pressure gradients applied.")
        else:
            obj.addvar("ldp", 0)

        if obj.ltempeq == 0 or (obj.iwalltemp == 1 and obj.iwallmom == 2):
            obj.iwallmom = 3

        # &INPS
        obj.addvar("zsize", 96)
        obj.addvar("lzstretch", 0)
        obj.addvar("stl_file", "")
        obj.addvar("gen_geom", True)
        obj.addvar("geom_path", "")
        obj.addvar("diag_neighbs", True)
        obj.addvar("stl_ground", True)

        if obj.lzstretch:
            obj.addvar("stretchconst", 0.01)
            obj.addvar("lstretchexp", 0)
            obj.addvar("lstretchexpcheck", 0)
            obj.addvar("lstretchtanh", 0)
            obj.addvar("lstretch2tanh", 0)
            obj.addvar("hlin", 0)
            obj.addvar("dzlin", 0)
            obj.addvar("dz", obj.dzlin)
        else:
            obj.addvar("dz", obj.zsize / obj.ktot)

        if obj.lEB:
            obj.addvar("maxlen", 10)
        else:
            obj.addvar("maxlen", np.inf)

        obj.addvar("u0", 0)
        obj.addvar("v0", 0)
        obj.addvar("tke", 0)
        obj.addvar("dpdx", 0)
        obj.addvar("dpdy", 0)
        obj.addvar("thl0", 288)
        obj.addvar("qt0", 0)

        obj.addvar("nsv", 0)
        if obj.nsv > 0:
            obj.addvar("sv10", 0)
            obj.addvar("sv20", 0)
            obj.addvar("sv30", 0)
            obj.addvar("sv40", 0)
            obj.addvar("sv50", 0)
            obj.addvar("lscasrc", 0)
            obj.addvar("lscasrcl", 0)
            obj.addvar("lscasrcr", 0)
            obj.addvar("xS", -1)
            obj.addvar("yS", -1)
            obj.addvar("zS", -1)
            obj.addvar("SSp", -1)
            obj.addvar("sigSp", -1)
            obj.addvar("nscasrc", 0)
            obj.addvar("xSb", -1)
            obj.addvar("ySb", -1)
            obj.addvar("zSb", -1)
            obj.addvar("xSe", -1)
            obj.addvar("ySe", -1)
            obj.addvar("zSe", -1)
            obj.addvar("SSl", -1)
            obj.addvar("sigSl", -1)
            obj.addvar("nscasrcl", 0)

        obj.addvar("lapse", 0)
        obj.addvar("w_s", 0)
        obj.addvar("R", 0)

        obj.addvar("libm", 1)

        obj.addvar("isolid_bound", 1)
        obj.addvar("ifacsec", 1)

        obj.addvar("read_types", 0)
        if obj.read_types:
            obj.addvar("types_path", "")

        if obj.lEB:
            obj.addvar("xazimuth", 90)
            obj.addvar("ltimedepsw", 0)
            obj.addvar("ishortwave", 1)
            obj.addvar("isolar", 1)
            obj.addvar("runtime", 0)
            obj.addvar("dtEB", 10.0)
            obj.addvar("dtSP", obj.dtEB)

            if obj.isolar == 1:
                obj.addvar("solarazimuth", 135)
                obj.addvar("solarzenith", 28.4066)
                obj.addvar("I", 800)
                obj.addvar("Dsky", 418.8041)
            elif obj.isolar == 2:
                obj.addvar("longitude", -0.13)
                obj.addvar("latitude", 51.5)
                obj.addvar("timezone", 0)
                obj.addvar("elevation", 0)
                obj.addvar("hour", 6)
                obj.addvar("minute", 0)
                obj.addvar("second", 0)
                obj.addvar("year", 2011)
                obj.addvar("month", 9)
                obj.addvar("day", 30)
            elif obj.isolar == 3:
                obj.addvar("weatherfname", "")
                obj.addvar("hour", 0)
                obj.addvar("minute", 0)
                obj.addvar("second", 0)
                obj.addvar("year", 0)
                obj.addvar("month", 6)
                obj.addvar("day", 1)

            obj.addvar("psc_res", 0.1)
            obj.addvar("lvfsparse", False)

            obj.addvar("calc_vf", True)
            obj.addvar("maxD", np.inf)

            if not obj.calc_vf:
                obj.addvar("vf_path", "")

            obj.addvar("view3d_out", 0)
            if obj.view3d_out == 2 and not obj.lvfsparse:
                raise ValueError("If sparse view3d output is desired, set lvfsparse=.true. in &ENERGYBALANCE.")

        obj.addvar("facT", 288.0)
        obj.addvar("nfaclyrs", 3)
        obj.addvar("nfcts", 0)
        Preprocessing.generate_factypes(obj)
        obj.addvar("facT_file", "")

    @staticmethod
    def generate_factypes(obj: "Preprocessing") -> None:
        """Generate factypes array."""
        K = obj.nfaclyrs
        factypes = []

        # Bounding walls (bw)
        id_bw = -101
        lGR_bw = 0
        z0_bw = 0
        z0h_bw = 0
        al_bw = 0.5
        em_bw = 0.85
        D_bw = 0.0
        d_bw = D_bw / K
        C_bw = 0.0
        l_bw = 0.0
        k_bw = 0.0
        bw = [
            id_bw,
            lGR_bw,
            z0_bw,
            z0h_bw,
            al_bw,
            em_bw,
        ] + [d_bw] * K + [C_bw] * K + [l_bw] * K + [k_bw] * (K + 1)
        factypes.append(bw)

        # Floors (f)
        id_f = -1
        lGR_f = 0
        z0_f = 0.05
        z0h_f = 0.00035
        al_f = 0.5
        em_f = 0.85
        D_f = 0.5
        d_f = D_f / K
        C_f = 1.875e6
        l_f = 0.75
        k_f = 0.4e-6
        if K == 3:
            f = [
                id_f,
                lGR_f,
                z0_f,
                z0h_f,
                al_f,
                em_f,
                0.1,
                0.2,
                0.2,
            ] + [C_f] * K + [l_f] * K + [k_f] * (K + 1)
        else:
            f = [
                id_f,
                lGR_f,
                z0_f,
                z0h_f,
                al_f,
                em_f,
            ] + [d_f] * K + [C_f] * K + [l_f] * K + [k_f] * (K + 1)
        factypes.append(f)

        # Dummy (dm)
        id_dm = 0
        lGR_dm = 0
        z0_dm = 0
        z0h_dm = 0
        al_dm = 0
        em_dm = 0
        D_dm = 0.3
        d_dm = D_dm / K
        C_dm = 1.875e6
        l_dm = 0.75
        k_dm = 0.4e-6
        dm = [
            id_dm,
            lGR_dm,
            z0_dm,
            z0h_dm,
            al_dm,
            em_dm,
        ] + [d_dm] * K + [C_dm] * K + [l_dm] * K + [k_dm] * (K + 1)
        factypes.append(dm)

        # Concrete (c)
        id_c = 1
        lGR_c = 0
        z0_c = 0.05
        z0h_c = 0.00035
        al_c = 0.5
        em_c = 0.85
        D_c = 0.36
        d_c = D_c / K
        C_c = 2.5e6
        l_c = 1
        k_c = 0.4e-6
        c = [
            id_c,
            lGR_c,
            z0_c,
            z0h_c,
            al_c,
            em_c,
        ] + [d_c] * K + [C_c] * K + [l_c] * K + [k_c] * (K + 1)
        factypes.append(c)

        # Brick (b)
        id_b = 2
        lGR_b = 0
        z0_b = 0.05
        z0h_b = 0.00035
        al_b = 0.5
        em_b = 0.85
        D_b = 0.36
        d_b = D_b / K
        C_b = 2.766667e6
        l_b = 0.83
        k_b = 0.3e-6
        b = [
            id_b,
            lGR_b,
            z0_b,
            z0h_b,
            al_b,
            em_b,
        ] + [d_b] * K + [C_b] * K + [l_b] * K + [k_b] * (K + 1)
        factypes.append(b)

        # Stone (s)
        id_s = 3
        lGR_s = 0
        z0_s = 0.05
        z0h_s = 0.00035
        al_s = 0.5
        em_s = 0.85
        D_s = 0.36
        d_s = D_s / K
        C_s = 2.19e6
        l_s = 2.19
        k_s = 1e-6
        s = [
            id_s,
            lGR_s,
            z0_s,
            z0h_s,
            al_s,
            em_s,
        ] + [d_s] * K + [C_s] * K + [l_s] * K + [k_s] * (K + 1)
        factypes.append(s)

        # Wood (w)
        id_w = 4
        lGR_w = 0
        z0_w = 0.05
        z0h_w = 0.00035
        al_w = 0.5
        em_w = 0.85
        D_w = 0.36
        d_w = D_w / K
        C_w = 1e6
        l_w = 0.1
        k_w = 0.1e-6
        w = [
            id_w,
            lGR_w,
            z0_w,
            z0h_w,
            al_w,
            em_w,
        ] + [d_w] * K + [C_w] * K + [l_w] * K + [k_w] * (K + 1)
        factypes.append(w)

        # GR1
        id_GR1 = 11
        lGR_GR1 = 1
        z0_GR1 = 0.05
        z0h_GR1 = 0.00035
        al_GR1 = 0.25
        em_GR1 = 0.95
        D_GR1 = 0.6
        d_GR1 = D_GR1 / K
        C_GR1 = 5e6
        l_GR1 = 2
        k_GR1 = 0.4e-6
        GR1 = [
            id_GR1,
            lGR_GR1,
            z0_GR1,
            z0h_GR1,
            al_GR1,
            em_GR1,
        ] + [d_GR1] * K + [C_GR1] * K + [l_GR1] * K + [k_GR1] * (K + 1)
        factypes.append(GR1)

        # GR2
        id_GR2 = 12
        lGR_GR2 = 1
        z0_GR2 = 0.05
        z0h_GR2 = 0.00035
        al_GR2 = 0.35
        em_GR2 = 0.90
        D_GR2 = 0.6
        d_GR2 = D_GR2 / K
        C_GR2 = 2e6
        l_GR2 = 0.8
        k_GR2 = 0.4e-6
        GR2 = [
            id_GR2,
            lGR_GR2,
            z0_GR2,
            z0h_GR2,
            al_GR2,
            em_GR2,
        ] + [d_GR2] * K + [C_GR2] * K + [l_GR2] * K + [k_GR2] * (K + 1)
        factypes.append(GR2)

        obj.addvar("factypes", np.array(factypes))

    @staticmethod
    def generate_xygrid(obj: "Preprocessing") -> None:
        """Generate x and y grids."""
        obj.addvar("xf", np.arange(0.5 * obj.dx, obj.xlen, obj.dx))
        obj.addvar("yf", np.arange(0.5 * obj.dy, obj.ylen, obj.dy))
        obj.addvar("xh", np.arange(0, obj.xlen + obj.dx, obj.dx))
        obj.addvar("yh", np.arange(0, obj.ylen + obj.dy, obj.dy))

    @staticmethod
    def generate_zgrid(obj: "Preprocessing") -> None:
        """Generate z grid."""
        if not obj.lzstretch:
            obj.addvar("zf", np.arange(0.5 * obj.dz, obj.zsize, obj.dz))
            obj.addvar("zh", np.arange(0, obj.zsize + obj.dz, obj.dz))
            obj.addvar("dzf", obj.zh[1:] - obj.zh[:-1])
        else:
            raise NotImplementedError("Stretched z-grid not yet implemented in Python")

    @staticmethod
    def generate_lscale(obj: "Preprocessing") -> None:
        """Generate lscale array."""
        if (
            (obj.luoutflowr or obj.lvoutflowr)
            + (obj.luvolflowr or obj.lvvolflowr)
            + obj.lprofforc
            + obj.lcoriol
            + obj.ldp
        ) > 1:
            raise ValueError("More than one forcing type specified")

        ls = np.zeros((len(obj.zf), 10))
        ls[:, 0] = obj.zf
        ls[:, 5] = obj.w_s
        ls[:, 9] = obj.R
        if obj.lprofforc or obj.lcoriol:
            ls[:, 1] = obj.u0
            ls[:, 2] = obj.v0
        elif obj.ldp:
            ls[:, 3] = obj.dpdx
            ls[:, 4] = obj.dpdy

        obj.addvar("ls", ls)

    @staticmethod
    def write_lscale(obj: "Preprocessing") -> None:
        """Write lscale.inp.* file."""
        fname = f"lscale.inp.{obj.expnr}"
        with open(fname, "w") as f:
            # MATLAB: fprintf(lscale, '%-12s\n', '# SDBL flow');
            # MATLAB: fprintf(lscale, '%-60s\n', '# z uq vq pqx pqy wfls dqtdxls dqtdyls dqtdtls dthlrad');
            f.write("# SDBL flow \n")
            f.write("# z uq vq pqx pqy wfls dqtdxls dqtdyls dqtdtls dthlrad      \n")
            # MATLAB: fprintf(lscale, '%-20.15f %-12.6f %-12.6f %-12.9f %-12.9f %-15.9f %-12.6f %-12.6f %-12.6f %-17.12f\n', obj.ls');
            # Left-aligned format
            for row in obj.ls:
                f.write(
                    f"{row[0]:<20.15f} {row[1]:<12.6f} {row[2]:<12.6f} "
                    f"{row[3]:<12.9f} {row[4]:<12.9f} {row[5]:<15.9f} "
                    f"{row[6]:<12.6f} {row[7]:<12.6f} {row[8]:<12.6f} {row[9]:<17.12f}\n"
                )

    @staticmethod
    def generate_prof(obj: "Preprocessing") -> None:
        """Generate prof array."""
        pr = np.zeros((len(obj.zf), 6))
        pr[:, 0] = obj.zf

        if obj.lapse:
            thl = np.zeros(obj.ktot)
            thl[0] = obj.thl0
            for k in range(obj.ktot - 1):
                thl[k + 1] = thl[k] + obj.lapse * obj.zsize / obj.ktot
            pr[:, 1] = thl
        else:
            pr[:, 1] = obj.thl0

        pr[:, 2] = obj.qt0
        pr[:, 3] = obj.u0
        pr[:, 4] = obj.v0
        pr[:, 5] = obj.tke

        obj.addvar("pr", pr)

    @staticmethod
    def write_prof(obj: "Preprocessing") -> None:
        """Write prof.inp.* file."""
        fname = f"prof.inp.{obj.expnr}"
        with open(fname, "w") as f:
            # MATLAB: fprintf(prof, '%-12s\n', '# SDBL flow');
            # MATLAB: fprintf(prof, '%-60s\n', '# z thl qt u v tke');
            f.write("# SDBL flow \n")
            f.write("# z thl qt u v tke                                          \n")
            # MATLAB: fprintf(prof, '%-20.15f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f\n', obj.pr');
            # Left-aligned format
            for row in obj.pr:
                f.write(
                    f"{row[0]:<20.15f} {row[1]:<12.6f} {row[2]:<12.6f} "
                    f"{row[3]:<12.6f} {row[4]:<12.6f} {row[5]:<12.6f}\n"
                )

    @staticmethod
    def generate_scalar(obj: "Preprocessing") -> None:
        """Generate scalar array."""
        sc = np.zeros((len(obj.zf), obj.nsv + 1))
        sc[:, 0] = obj.zf
        if obj.nsv > 0:
            sc[:, 1] = obj.sv10
        if obj.nsv > 1:
            sc[:, 2] = obj.sv20
        if obj.nsv > 2:
            sc[:, 3] = obj.sv30
        if obj.nsv > 3:
            sc[:, 4] = obj.sv40
        if obj.nsv > 4:
            sc[:, 5] = obj.sv50

        obj.addvar("sc", sc)

    @staticmethod
    def write_scalar(obj: "Preprocessing") -> None:
        """Write scalar.inp.* file."""
        fname = f"scalar.inp.{obj.expnr}"
        with open(fname, "w") as f:
            f.write("# SDBL flow\n")
            f.write("# z scaN,  N=1,2...nsv\n")
            for row in obj.sc:
                f.write(f"{row[0]:20.15f}")
                for i in range(1, obj.nsv + 1):
                    f.write(f" {row[i]:14.10f}")
                f.write("\n")

    @staticmethod
    def generate_scalarsources(obj: "Preprocessing") -> None:
        """Generate scalar sources."""
        if (
            obj.lscasrc
            and obj.nscasrc < 2
            and any(
                [
                    obj.nsv == 0,
                    obj.nscasrc < 1,
                    obj.xS == -1,
                    obj.yS == -1,
                    obj.zS == -1,
                    obj.SSp == -1,
                    obj.sigSp == -1,
                ]
            )
        ):
            raise ValueError(
                "Must set non-zero positive nsv and nscasrc under &SCALARS, "
                "and appropriate xS, yS, zS, SSp and sigSp under &INPS for scalar point source"
            )

        if obj.lscasrc:
            scasrcp = np.zeros((obj.nscasrc, 5))
            if obj.nscasrc == 1:
                scasrcp[0, 0] = obj.xS
                scasrcp[0, 1] = obj.yS
                scasrcp[0, 2] = obj.zS
                scasrcp[0, 3] = obj.SSp
                scasrcp[0, 4] = obj.sigSp
            obj.addvar("scasrcp", scasrcp)

        if obj.lscasrcl:
            scasrcl = np.zeros((obj.nscasrcl, 8))
            if obj.nscasrcl == 1:
                scasrcl[0, 0] = obj.xSb
                scasrcl[0, 1] = obj.ySb
                scasrcl[0, 2] = obj.zSb
                scasrcl[0, 3] = obj.xSe
                scasrcl[0, 4] = obj.ySe
                scasrcl[0, 5] = obj.zSe
                scasrcl[0, 6] = obj.SSl
                scasrcl[0, 7] = obj.sigSl
            obj.addvar("scasrcl", scasrcl)

    @staticmethod
    def write_scalarsources(obj: "Preprocessing") -> None:
        """Write scalar sources files."""
        for ii in range(1, obj.nsv + 1):
            if obj.lscasrc:
                fname = f"scalarsourcep.inp.{ii}.{obj.expnr}"
                with open(fname, "w") as f:
                    f.write("# Scalar point source data\n")
                    f.write("#xS yS zS SS sigS\n")
                    for row in obj.scasrcp:
                        f.write(
                            f"{row[0]:12.6f}\t {row[1]:12.6f}\t {row[2]:12.6f}\t "
                            f"{row[3]:12.6f}\t {row[4]:12.6f}\t\n"
                        )

            if obj.lscasrcl:
                fname = f"scalarsourcel.inp.{ii}.{obj.expnr}"
                with open(fname, "w") as f:
                    f.write("# Scalar line source data\n")
                    f.write("#xSb ySb zSb xSe ySe zSe SS sigS\n")
                    for row in obj.scasrcl:
                        f.write(
                            f"{row[0]:12.6f}\t {row[1]:12.6f}\t {row[2]:12.6f}\t "
                            f"{row[3]:12.6f}\t {row[4]:12.6f}\t {row[5]:12.6f}\t "
                            f"{row[6]:12.6f}\t {row[7]:12.6f}\t\n"
                        )

    @staticmethod
    def set_nfcts(obj: "Preprocessing", nfcts: int) -> None:
        """Set number of facets."""
        obj.nfcts = nfcts

    @staticmethod
    def write_facets(obj: "Preprocessing", types: np.ndarray, normals: np.ndarray) -> None:
        """Write facets.inp.* file."""
        fname = f"facets.inp.{obj.expnr}"
        with open(fname, "w") as f:
            f.write("# type, normal\n")
            # MATLAB format: fprintf(fileID, '%-4d %-4.4f %-4.4f %-4.4f\n', [types normals]');
            # %-4d means left-align integer in 4 characters (e.g., "1   " for 1)
            # %-4.4f means left-align float with 4 total width, 4 decimals (e.g., "0.0000")
            for i in range(len(types)):
                # Format integer left-aligned in 4 chars: "1   " not "   1"
                type_val = int(types[i])
                type_str = f"{type_val:<4d}"  # Left-align: "1   "
                # Format floats left-aligned: "0.0000" not " 0.0000"
                f.write(f"{type_str} {normals[i,0]:<4.4f} {normals[i,1]:<4.4f} {normals[i,2]:<4.4f}\n")

    @staticmethod
    def write_factypes(obj: "Preprocessing") -> None:
        """Write factypes.inp.* file."""
        K = obj.nfaclyrs
        fname = f"factypes.inp.{obj.expnr}"

        with open(fname, "w") as f:
            f.write(f"# walltype, {K} layers per type where layer 1 is the outdoor side and layer {K} is indoor side\n")
            f.write("# 0=default dummy, -1=asphalt floors;-101=concrete bounding walls;1=concrete;2=bricks;3=stone;4=painted wood;11=GR1; 12=GR2\n")
            f.write("# wallid  lGR  z0 [m]  z0h [m]  al [-]  em [-]")
            for k in range(1, K + 1):
                f.write(f"  d{k} [m]")
            for k in range(1, K + 1):
                f.write(f"  C{k} [J/(K m^3)]")
            for k in range(1, K + 1):
                f.write(f"  l{k} [W/(m K)]")
            for k in range(1, K + 2):
                f.write(f"  k{k} [W/(m K)]")
            f.write("\n")

            for row in obj.factypes:
                f.write(f"{int(row[0]):8d}  {int(row[1]):3d}  {row[2]:6.2f}  {row[3]:7.5f}  {row[4]:6.2f}  {row[5]:6.2f}")
                for k in range(K):
                    f.write(f"  {row[6+k]:6.2f}")
                for k in range(K):
                    f.write(f"  {row[6+K+k]:14.0f}")
                for k in range(K):
                    f.write(f" {row[6+2*K+k]:13.4f}")
                for k in range(K + 1):
                    f.write(f" {row[6+3*K+k]:13.8f}")
                f.write("\n")

    @staticmethod
    def generate_albedos(obj: "Preprocessing", facet_types: np.ndarray) -> np.ndarray:
        """Generate albedos array."""
        typeids = obj.factypes[:, 0]
        albedos = []
        for i in range(obj.nfcts):
            my_typid = facet_types[i]
            idx = np.where(typeids == my_typid)[0]
            if len(idx) > 0:
                albedo = obj.factypes[idx[0], 4]
                albedos.append(albedo)
            else:
                albedos.append(0.0)
        return np.array(albedos)

    @staticmethod
    def write_facetarea(obj: "Preprocessing", facetarea: np.ndarray) -> None:
        """Write facetarea.inp.* file."""
        fname = f"facetarea.inp.{obj.expnr}"
        with open(fname, "w") as f:
            f.write("# area of facets\n")
        with open(fname, "ab") as f:
            np.savetxt(f, facetarea, fmt="%4f", delimiter=" ")

    @staticmethod
    def write_svf(obj: "Preprocessing", svf: np.ndarray) -> None:
        """Write svf.inp.* file."""
        fname = f"svf.inp.{obj.expnr}"
        with open(fname, "w") as f:
            f.write("# sky view factors\n")
        with open(fname, "ab") as f:
            np.savetxt(f, svf, fmt="%4f", delimiter=" ")

    @staticmethod
    def write_netsw(obj: "Preprocessing", Knet: np.ndarray) -> None:
        """Write netsw.inp.* file."""
        fname = f"netsw.inp.{obj.expnr}"
        with open(fname, "w") as f:
            f.write("# net shortwave on facets [W/m2] (including reflections and diffusive)\n")
            for val in Knet:
                f.write(f"{val:6.4f}\n")

    @staticmethod
    def write_timedepsw(obj: "Preprocessing", tSP: np.ndarray, Knet: np.ndarray) -> None:
        """Write timedepsw.inp.* file."""
        fname = f"timedepsw.inp.{obj.expnr}"
        with open(fname, "w") as f:
            f.write("# time-dependent net shortwave on facets [W/m2]. First line: times (1 x nt), then netsw (nfcts x nt)\n")
        with open(fname, "ab") as f:
            np.savetxt(f, tSP, fmt="%9.2f", delimiter=" ")
            np.savetxt(f, Knet, fmt="%9.4f", delimiter=" ")

    @staticmethod
    def write_Tfacinit(obj: "Preprocessing", Tfacinit: np.ndarray) -> None:
        """Write Tfacinit.inp.* file."""
        fname = f"Tfacinit.inp.{obj.expnr}"
        with open(fname, "w") as f:
            f.write("# Initial facet tempereatures in radiative equilibrium\n")
        with open(fname, "ab") as f:
            np.savetxt(f, Tfacinit, fmt="%4f", delimiter=" ")

    @staticmethod
    def write_Tfacinit_layers(obj: "Preprocessing", Tfacinit_layers: np.ndarray) -> None:
        """Write Tfacinit_layers.inp.* file."""
        fname = f"Tfacinit_layers.inp.{obj.expnr}"
        with open(fname, "w") as f:
            f.write("# Initial facet tempereatures in radiative equilibrium\n")
        with open(fname, "ab") as f:
            np.savetxt(f, Tfacinit_layers, fmt="%4f", delimiter=" ")

    @staticmethod
    def update_namoptions(namoptionsfile: str, sectionname: str, varname: str, value: int) -> None:
        """Update namoptions file with new value."""
        # MATLAB version: uses regexprep to replace existing pattern
        # pattern = [varname ' * = * \d+'];
        # new_content = regexprep(namoptions_content, pattern, sprintf('%s = %d', varname, value));
        with open(namoptionsfile, "r") as f:
            content = f.read()

        # Pattern to match: varname = number (with optional whitespace)
        # Use word boundary to match whole word only
        pattern = rf"\b{re.escape(varname)}\s*=\s*\d+"
        
        # Check if variable exists in file (using word boundary)
        if re.search(rf"\b{re.escape(varname)}\b", content):
            # Replace existing value
            new_content = re.sub(pattern, f"{varname} = {value}", content, flags=re.IGNORECASE)
        elif sectionname in content:
            # Variable doesn't exist, but section does - add it after section name
            new_content = re.sub(
                sectionname,
                f"{sectionname}\n{varname} = {value}",
                content,
                count=1  # Only replace first occurrence
            )
        else:
            # Section doesn't exist - add section and variable
            content = content + f"\n{sectionname}\n"
            new_content = re.sub(
                sectionname,
                f"{sectionname}\n{varname} = {value}",
                content,
                count=1
            )
            new_content = new_content + "\n/"

        with open(namoptionsfile, "w") as f:
            f.write(new_content)

