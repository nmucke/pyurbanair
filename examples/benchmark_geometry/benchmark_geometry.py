import numpy as np

import typer
from benchmark_geometry_utils import Building


class XieCastroBenchmarkGeometry:
    """class representing the XieCastroBenchmarkGeometry"""

    def __init__(
        self,
        heights: np.array = np.array(
            [
                [10, 13.6, 10, 2.8],
                [13.6, 10, 6.4, 10],
                [13.6, 6.4, 17.2, 10],
                [10, 6.4, 13.6, 10],
            ]
        ),
        num_tiles: tuple[int, int] = (2, 2),
        resolution_factor: int = 2,
    ):
        """
        Docstring für __init__

                    :param heights: The heights of the individual buildings  defaults to the Xie Castro setup
            :param resolution_factor: Multiple of the default resolution (which is [8,8,10] * average building height)
        :type resolution_factor: int
        """

        assert np.asarray(heights).shape == (
            4,
            4,
        ), "The Xie Castro Benchmark needs 4x4 building heights"
        self.heights = heights
        self.height_mean = np.mean(self.heights)
        self.box_size = self.height_mean
        self.intrinsic_size = np.array([8, 8, 10], dtype=int)
        self.lower = np.array([0, 0, 0])
        self.upper = self.box_size * self.intrinsic_size * np.append(num_tiles, 1)
        self.tile_size = self.box_size * self.intrinsic_size
        self.num_cells = (
            resolution_factor * 2 * self.intrinsic_size * np.append(num_tiles, 1)
        )
        self.num_tiles = num_tiles
        self.buildings = self._create_building_list()

    def _create_building_list(self) -> list[Building]:
        box_size = self.box_size
        ro = []
        for i in range(self.num_tiles[0]):
            ro.append([i * self.tile_size[0] + box_size / 2, box_size / 2])
            ro.append([i * self.tile_size[0] + 5 * box_size / 2, 3 * box_size / 2])
            ro.append([i * self.tile_size[0] + 9 * box_size / 2, box_size / 2])
            ro.append([i * self.tile_size[0] + 13 * box_size / 2, 3 * box_size / 2])

        row_origins = np.array(ro)

        buildings = []
        tiles = self.num_tiles
        tile_size = self.tile_size
        tiled_heights = np.tile(self.heights, tiles)
        # aligned rows
        # idx = [0, 2]
        idx = np.arange(0, len(row_origins), 2)
        for origin, hs in zip(row_origins[idx, :], tiled_heights[idx, :]):
            for i, h in enumerate(hs):
                buildings.append(
                    Building(
                        np.array([0, 2 * i * box_size]) + origin,
                        size=[box_size, box_size],
                        height=h,
                    )
                )

        # shifted rows
        # idx = [1, 3]
        idx = np.arange(1, len(row_origins), 2)
        for origin, hs in zip(row_origins[idx, :], tiled_heights[idx, :]):
            for i, h in enumerate(hs[:-1]):
                buildings.append(
                    Building(
                        np.array([0, 2 * i * box_size]) + origin,
                        size=[box_size, box_size],
                        height=h,
                    )
                )

            buildings.append(
                Building(
                    np.array([0, -3 * box_size / 2]) + origin,
                    size=[box_size, box_size / 2],
                    height=hs[-1],
                )
            )
            print((tiles[1] - 1) * tile_size[1] + box_size * 6 * box_size + origin)
            buildings.append(
                Building(
                    np.array([0, (tiles[1] - 1) * tile_size[1] + 6 * box_size])
                    + origin,
                    size=[box_size, box_size / 2],
                    height=hs[-1],
                )
            )
        return buildings

    def _to_uniform_cartesian(self) -> tuple[np.array, np.array]:
        xs = np.linspace(self.lower[0], self.upper[0], self.num_cells[0] + 1)
        ys = np.linspace(self.lower[1], self.upper[1], self.num_cells[1] + 1)
        zs = np.linspace(self.lower[2], self.upper[2], self.num_cells[2] + 1)
        grid = np.array(np.meshgrid(xs, ys, zs))
        mask = np.zeros(
            shape=(self.num_cells[0], self.num_cells[1], self.num_cells[2]), dtype=bool
        )
        indices = []
        for b in self.buildings:
            print(b.bounding_box)
            mask |= b.compute_mask(grid)
            indices.append(b.compute_index(xs, ys, zs))
        return mask, np.array(indices)

    def to_lbm(self) -> str:
        _, indices = self._to_uniform_cartesian()
        indices = indices.reshape(-1, 6)
        output = (
            "module m_city3\n"
            "contains\n"
            "subroutine city3(blanking)\n"
            "use mod_dimensions, only : nx, nyg, nz\n"
            "implicit none\n"
            "logical, intent(inout) :: blanking(0:nx+1,0:nyg+1,0:nz+1)\n"
            "integer ioff\n"
            "integer joff\n\n"
            "ioff=10\n"
            "joff=0\n"
        )
        max_num_digit = max(len(s) for s in self.num_cells.astype(str))
        mnd = max_num_digit
        print(self.num_cells)
        for row in indices:
            idx = row + 1
            output += f"blanking(ioff+{idx[0]:{mnd}}:ioff+{idx[3]:{mnd}}, joff+{idx[1]:{mnd}}:joff+{idx[4]:{mnd}}, {idx[2]:{mnd}}:{idx[5]:{mnd}})=.true.\n"
        output += "end subroutine\nend module"
        return output

    def to_stl(self) -> str:
        return "\n".join([b.to_stl() for b in self.buildings])


def test_BenchmarkGeometry() -> None:
    """just some plotting and writing"""
    import matplotlib.pyplot as plt

    geometry = XieCastroBenchmarkGeometry()

    mask, _ = geometry._to_uniform_cartesian()

    x = np.arange(mask.shape[0])
    y = np.arange(mask.shape[1])
    plt.pcolormesh(x, y, np.any(mask, axis=2), edgecolors="k", linewidth=0.5)
    plt.gca().set_aspect("equal")
    plt.show()
    x, y, z = np.indices(np.array(mask.shape) + 1).astype(float)
    x -= 0.5
    y -= 0.5
    z -= 0.5

    fig = plt.axes(projection="3d")

    ax_vox = fig.voxels(x, y, z, mask, edgecolor="gray", linewidth=0.5)
    plt.show()
    with open("test.stl", "w") as f:
        f.write(geometry.to_stl())
    with open("test.f90", "w") as f:
        f.write(geometry.to_lbm())


def main(output_type: str, filename: str, resolution: int = 1) -> None:
    """
    Command Line Tool to generate the geometry for the Xie Castro benchmark.

    :param output_type: Choose either stl (for OpenFoam), lbm (for Geir Evensen's code or palm for the Palm model system)
    :type output_type: str
    :param filename: Filename the output is written to (do not include the extension)
    :type filename: str
    :param resolution: multiple of the default resolution of the simulation
    :type resolution: int
    """
    if output_type not in ["lbm", "stl", "palm"]:
        print("Invalid type!")
        raise typer.Exit()
    geometry = XieCastroBenchmarkGeometry(resolution_factor=resolution)
    match output_type:
        case "lbm":
            with open(filename + ".ini", "w") as f:
                f.write(geometry.to_lbm())
        case "stl":
            with open(filename + ".stl", "w") as f:
                f.write(geometry.to_stl())
        case "palm":
            raise NotImplementedError("Palm output not implemented yet")
            # with open(filename + ".palm", "w") as f:
            #    f.write(geometry.to_palm())
        case _:
            pass


if __name__ == "__main__":
    test_BenchmarkGeometry()
    typer.run(main)
