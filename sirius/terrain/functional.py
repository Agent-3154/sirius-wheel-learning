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


def platform_with_stairs(
    size: tuple[float, float],
    num_steps: int = 5,
    step_width: float = 0.15,
    step_height: float = 0.1
):
    platform_width = size[1] * 0.4 - num_steps * step_width
    if platform_width < 0:
        raise ValueError("The platform width is too large for the given number of steps and step width.")
    
    meshes = []
    ground = trimesh.creation.box([size[0], size[1], 0.1])
    ground.apply_translation([0.0, 0.0, -0.05])
    meshes.append(ground)
    for i in range(num_steps):
        height = (i+1) * step_height
        box = trimesh.creation.box([size[0], step_width, height])
        box.apply_translation([0, step_width * i, height / 2])
        meshes.append(box)
    
    platform = trimesh.creation.box([size[0], platform_width, height])
    platform.apply_translation([0.0, i* step_width + platform_width / 2, height / 2])
    meshes.append(platform)
    mesh: trimesh.Trimesh = trimesh.util.concatenate(meshes)
    mesh.merge_vertices()
    mesh.apply_translation([size[0] / 2, size[1] / 2, 0.0])
    origin = np.array([size[0] / 2, size[1] / 4, 0.0])
    return [mesh], origin


def pallet_with_platform(
    size: tuple[float, float],
    platform_width: float,
    board_width: float,
    num_stringers: int,
    interval: float = 0.1,
    rotate: bool = False
):
    meshes = []
    board_thickness = 0.05
    if rotate:
        num_boards = int(size[0] / (board_width + interval))
        for i in range(num_boards):
            board = trimesh.creation.box([size[0], board_width, board_thickness])
            board.apply_translation([0.0, i * (board_width + interval), -board_thickness / 2])
            board.apply_translation([0.0, (board_width + interval) / 2 - size[1] / 2, 0.0])
            meshes.append(board)
        stringer_interval = size[1] / (num_stringers + 1)
        for i in range(1, num_stringers + 1):
            stringer = trimesh.creation.box([0.1, size[1], board_thickness])
            stringer.apply_translation([i * stringer_interval, 0.0, -board_thickness])
            stringer.apply_translation([-size[0] / 2, 0.0, 0.0])
            meshes.append(stringer)
    else:
        num_boards = int(size[1] / (board_width + interval))
        for i in range(num_boards):
            board = trimesh.creation.box([board_width, size[1], board_thickness])
            board.apply_translation([i * (board_width + interval), 0.0, -board_thickness / 2])
            board.apply_translation([(board_width + interval) / 2 - size[0] / 2, 0.0, 0.0])
            meshes.append(board)
        stringer_interval = size[0] / (num_stringers + 1)
        for i in range(1, num_stringers + 1):
            stringer = trimesh.creation.box([size[0], 0.1, board_thickness])
            stringer.apply_translation([0.0, i * stringer_interval, -board_thickness])
            stringer.apply_translation([0.0, -size[1] / 2, 0.0])
            meshes.append(stringer)
    platform = trimesh.creation.box([platform_width, platform_width, 0.05])
    meshes.append(platform)
    mesh: trimesh.Trimesh = trimesh.util.concatenate(meshes)
    mesh.merge_vertices()
    mesh.apply_translation([size[0] / 2, size[1] / 2, 0.0])
    origin = np.array([size[0] / 2, size[1] / 2, 0.025])
    return [mesh], origin


if __name__ == "__main__":
    # meshes = platform_with_slope((8.0, 8.0), 0.4)
    meshes = pallet_with_platform(
        (8.0, 8.0),
        platform_width=2.0,
        board_width=0.2,
        interval=0.1,
        num_stringers=4,
        rotate=True
    )
    scene = trimesh.Scene(meshes)
    scene.show()

