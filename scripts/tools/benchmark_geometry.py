"""Generate the Xie & Castro (2008) benchmark geometry as an STL or an LBM geometry file.

Usage:
    pixi run -e dev python scripts/tools/benchmark_geometry.py stl <name> \
        [--resolution N] [--num-tiles NX NY]

Writes ``<name>.stl`` (or ``<name>.ini`` for ``lbm``) in the current directory.
The committed case STL is ``geometries/xie_and_castro/xie_castro_2008_STL.stl``.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import typer
from numpy import ndarray as Array


class Box(NamedTuple):
    lower: Array
    upper: Array

    def __call__(self, grid: Array) -> np.array:
        mask = (self.lower[:, None, None, None] < grid[:, :-1, :-1, :-1]) | np.isclose(
            self.lower[:, None, None, None], grid[:, :-1, :-1, :-1]
        )
        mask &= (grid[:, 1:, 1:, 1:] < self.upper[:, None, None, None]) | np.isclose(
            grid[:, 1:, 1:, 1:], self.upper[:, None, None, None]
        )
        return mask


class Building:
    def __init__(
        self, lower: npt.ArrayLike, size: npt.ArrayLike, height: float, id: int = 0
    ):
        """
        Docstring für __init__

        :param lower: lower left (lower x,y) 2d coordinates of the building
        :type lower: npt.ArrayLike
        :param size: size of the building
        :type size: npt.ArrayLike
        :param height: building height
        :type height: float
        """

        upper = np.asarray(lower) + np.asarray(size)
        self.upper = np.append(upper, height)
        self.lower = np.append(lower, 0)
        self.coordinates = np.array(
            [
                [lower[0], lower[1]],
                [upper[0], lower[1]],
                [upper[0], upper[1]],
                [lower[0], upper[1]],
            ]
        )
        self.height = height
        self.id = id
        self.bounding_box = Box(self.lower, self.upper)

    def _to_triangles(self) -> np.array:
        vertices = np.concatenate(
            (
                np.column_stack((self.coordinates, np.full(len(self.coordinates), 0))),
                np.column_stack(
                    (self.coordinates, np.full(len(self.coordinates), self.height))
                ),
            )
        )
        triangles = []
        normals = []
        # lower normal to x
        triangles.append(np.array([vertices[0], vertices[1], vertices[4]]))
        triangles.append(np.array([vertices[1], vertices[5], vertices[4]]))
        normals.append(np.array([-1.0, 0, 0]))
        normals.append(np.array([-1.0, 0, 0]))
        # upper normal to y
        triangles.append(np.array([vertices[1], vertices[2], vertices[5]]))
        triangles.append(np.array([vertices[2], vertices[6], vertices[5]]))
        normals.append(np.array([0.0, 1.0, 0]))
        normals.append(np.array([0.0, 1.0, 0]))
        # upper normal to x
        triangles.append(np.array([vertices[2], vertices[3], vertices[6]]))
        triangles.append(np.array([vertices[3], vertices[7], vertices[6]]))
        normals.append(np.array([1.0, 0, 0]))
        normals.append(np.array([1.0, 0, 0]))
        # lower normal to y
        triangles.append(np.array([vertices[3], vertices[0], vertices[7]]))
        triangles.append(np.array([vertices[0], vertices[4], vertices[7]]))
        normals.append(np.array([0.0, -1.0, 0]))
        normals.append(np.array([0.0, -1.0, 0]))
        # upper normal to z
        triangles.append(np.array([vertices[4], vertices[5], vertices[7]]))
        triangles.append(np.array([vertices[5], vertices[6], vertices[7]]))
        normals.append(np.array([0.0, 0, 1.0]))
        normals.append(np.array([0.0, 0, 1.0]))
        # lower normal to z
        triangles.append(np.array([vertices[3], vertices[1], vertices[0]]))
        triangles.append(np.array([vertices[3], vertices[2], vertices[1]]))
        normals.append(np.array([0.0, 0, -1.0]))
        normals.append(np.array([0.0, 0, -1.0]))

        return np.array(triangles), np.array(normals)

    def to_stl(self) -> str:
        """
        Provide plain text stl to generate buildings geometry

        :param self:
        :return: The plain text stl
        :rtype: str
        """
        triangles, normals = self._to_triangles()

        stl = "solid building\n"
        for triangle, n in zip(triangles, normals):
            stl += "  facet normal " + f"{n[0]}" + f" {n[1]}" + f" {n[2]}\n"
            stl += "    outer loop\n"
            for v in triangle:
                stl += "      vertex" + f" {v[0]}" + f" {v[1]}" + f" {v[2]}\n"
            stl += "    endloop\n"
            stl += "  endfacet\n"
        stl += "endsolid building\n"
        return stl

    def compute_mask(self, grid: np.array) -> np.array:
        """
        Compute the bitmask (boolean) for a 3 dimensional cartesian uniform grid (all buildings starting at z=0)

        :param self:
        :param grid: grid of coordinates, shape = (3,num_cells_x+1,num_cells_y+1,num_cells_z+1)
                     see `np.meshgrid()`
        """
        return np.all(self.bounding_box(grid), axis=0)

    def compute_index(
        self, x_coords: np.array, y_coords: np.array, z_coords: np.array
    ) -> tuple[np.array, np.array]:
        """
        Docstring für compute_index

        :param self: Beschreibung
        :param x_coords: Array of x-axis coordinates, shape = (num_cells_x + 1,)
        :param y_coords: Array of y-axis coordinates, shape = (num_cells_y + 1,)
        :param z_coords: Array of z-axis coordinates, shape = (num_cells_z + 1,)
        """
        cell_boundaries = [x_coords, y_coords, z_coords]
        lower_idx = []
        for i in range(2):
            condition = self.lower[i] <= cell_boundaries[i]
            lower_idx.append(np.argmax(condition))

        condition = 0 <= cell_boundaries[2]
        lower_idx.append(np.argmax(condition))

        upper_idx = []
        for i in range(2):
            condition = self.upper[i] >= cell_boundaries[i]
            upper_idx.append(len(condition) - 1 - np.argmax(condition[::-1]) - 1)
        condition = self.height >= cell_boundaries[2]
        upper_idx.append(len(condition) - 1 - np.argmax(condition[::-1]) - 1)
        return np.array(lower_idx), np.array(upper_idx)


def display_building_list(
    buildings: list[Building], plot_transposed: bool = True
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    vl = []
    hs = []
    fig, ax = plt.subplots()

    # Create the collection
    for b in buildings:
        vl.append(b.coordinates)
        hs.append(b.height)
    vs = np.array(vl)

    if plot_transposed:
        vs = vs[..., ::-1]
    coll = PolyCollection(vs, edgecolors="black", alpha=0.6)
    coll.set_array(hs)
    ax.add_collection(coll)
    for v, h in zip(vs, hs):
        ax.text(*np.mean(v, axis=0), f"{h}", ha="center", va="center")
    ax.autoscale()

    if plot_transposed:
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_label_position("right")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    plt.show()


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


def main(
    output_type: str,
    filename: str,
    resolution: int = 1,
    num_tiles: tuple[int, int] = (2, 2),
) -> None:
    """
    Command Line Tool to generate the geometry for the Xie Castro benchmark.

    :param output_type: Choose either stl (for OpenFoam), lbm (for Geir Evensen's code or palm for the Palm model system)
    :type output_type: str
    :param filename: Filename the output is written to (do not include the extension)
    :type filename: str
    :param resolution: multiple of the default resolution of the simulation
    :type resolution: int
    """
    if output_type not in ["lbm", "stl"]:  # , "palm"]:
        print("Invalid type!")
        raise typer.Exit()
    geometry = XieCastroBenchmarkGeometry(
        resolution_factor=resolution, num_tiles=num_tiles
    )
    match output_type:
        case "lbm":
            with open(filename + ".ini", "w") as f:
                f.write(geometry.to_lbm())
        case "stl":
            with open(filename + ".stl", "w") as f:
                f.write(geometry.to_stl())
        case _:
            pass


if __name__ == "__main__":
    # test_BenchmarkGeometry()
    typer.run(main)
