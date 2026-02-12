from typing import NamedTuple

import numpy as np
import numpy.typing as npt
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
