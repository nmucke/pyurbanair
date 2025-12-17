import pathlib
import pdb

import matplotlib.pyplot as plt
import pylbm
import xarray

# from pylbm.compile_program import compile_lbm
from pylbm.forward_model import ForwardModel
from pylbm.stl_to_lbm import stl_to_lbm_geometry


def main() -> None:
    stl_path = pathlib.Path("examples/lbm/experiments/geom.STL")

    # stl_to_lbm_geometry(
    #     stl_path=stl_path,
    #     output_path=stl_path.parent / f"lol.F90",
    #     module_name="m_obstacle",
    #     subroutine_name="sphere",
    #     nx=200,
    #     ny=120,
    #     nz=96,
    # )

    forward_model = ForwardModel(
        stl_path=stl_path,
        nx=128,
        ny=128,
        nz=8,
    )
    forward_model.run()

    pdb.set_trace()

    state = xarray.load_dataset(".temp/lbm/fielddump.runcase.nc")
    plt.figure()
    plt.imshow(state.u.values[0, :, :, 100])
    plt.colorbar()
    plt.savefig("figures/u.png")


if __name__ == "__main__":
    main()
