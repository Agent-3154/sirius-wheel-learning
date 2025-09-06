import trimesh
import numpy as np


def platform_with_slope(size: tuple[float, float], height: float = 0.1):
    ground = trimesh.creation.box([size[0], size[1], 0.1]) # ground thickness is 0.1
    ground.apply_translation([0.0, 0.0, -0.05])
    platform = trimesh.creation.box([size[0], size[1] * 0.5, height])
    platform.apply_translation([0.0, 0.0, height * 0.5])
    slope = trimesh.creation.extrude_triangulation(
        vertices=np.array([[0.0, 0.0], [height, 0.0], [height, size[1] * 0.25]]),
        faces=np.array([[0, 1, 2]]),
        height=size[1]
    )
    slope.apply_transform(
        trimesh.transformations.rotation_matrix(
            angle=np.pi / 2,
            direction=np.array([0, 1, 0])
        )
    )
    slope.apply_translation([-size[0] * 0.5, size[1] * 0.25, height])
    mesh: trimesh.Trimesh = trimesh.util.concatenate([ground, platform, slope])
    mesh.merge_vertices()
    mesh.apply_translation([size[0] / 2, size[1] / 2, 0.0])
    origin = np.array([size[0] / 2, size[1] / 2, height])
    return [mesh], origin


if __name__ == "__main__":
    mesh = platform_with_slope((8.0, 8.0), 0.2)
    mesh.show()