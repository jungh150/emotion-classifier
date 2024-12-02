import numpy as np
import torch
import pyrender
import trimesh
from src.flame_pytorch.flame_pytorch import FLAME, get_config

def visualize_expression(expression_params):
    """
    FLAME 모델을 사용하여 3D 얼굴 시각화
    """
    # Config 설정
    config = get_config()
    config.batch_size = 1  # 배치 크기를 1로 설정
    flamelayer = FLAME(config)
    flamelayer.cuda()

    # 동일한 배치 크기
    batch_size = config.batch_size

    # 단일 샘플 입력
    shape_params = torch.zeros(batch_size, 100).cuda()
    pose_params_numpy = np.zeros((batch_size, 6), dtype=np.float32)
    pose_params_numpy[:, 0] = np.pi / 18  # X축 회전 (10도 숙임)
    pose_params_numpy[:, 1] = 0.0  # Y축 회전을 0으로 설정
    pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()

    # Forward Pass
    vertice, _ = flamelayer(shape_params, expression_params, pose_params)
    vertice = vertice.squeeze(0)
    faces = flamelayer.faces
    vertices = vertice.detach().cpu().numpy()
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

    # 3D 얼굴 시각화
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)

    # 카메라 추가
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],  # X축
        [0.0, 1.0, 0.0, 0.0],  # Y축
        [0.0, 0.0, 1.0, 0.5],  # Z축 (카메라 거리)
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    # 렌더링 및 시각화
    pyrender.Viewer(scene, use_raymond_lighting=True)
