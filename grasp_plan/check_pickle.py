from wrs import mgm, wd
import numpy as np
import pickle
import wrs.robot_sim.end_effectors.grippers.dh76.dh76 as dh
import wrs.modeling.collision_model as cm
import os
import wrs.basis.robot_math as rm

def show_pickle(base, pickle_path, obj_path=None, gripper=None, obj_rgba=None):
    """
    在仿真场景中加载并显示 pickle 文件中的抓取位姿，并可同时加载物体模型。

    参数:
        base: wd.World 实例（仿真场景）
        pickle_path: str, pickle 文件路径（如 'bottle_dh76.pickle'）
        obj_path: str, 可选，物体 STL 文件路径。若提供则加载并显示
        gripper: 夹爪实例，若为 None 则自动创建一个 Dh76 实例
        obj_rgba: list/np.array, 物体颜色和透明度，默认 [1,1,0,0.5]（黄色半透明）
    """
    # 1. 加载 pickle 文件
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"找不到文件: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        grasp_info_list = pickle.load(f)

    # 2. 加载物体模型（如果提供了路径）
    if obj_path is not None:
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"找不到物体文件: {obj_path}")
        object_model = cm.CollisionModel(obj_path)
        if obj_rgba is None:
            obj_rgba = rm.np.array([1, 1, 0, 0.5])  # 默认黄色半透明
        object_model.rgba = obj_rgba
        object_model.attach_to(base)

    # 3. 初始化夹爪（如果未传入）
    if gripper is None:
        gripper = dh.Dh76()

    # 4. 遍历每个抓取位姿，生成绿色半透明夹爪模型并附加
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        gripper.grip_at_by_pose(jaw_center_pos, jaw_center_rotmat, jaw_width)
        model = gripper.gen_meshmodel(rgb=[0, 1, 0], alpha=0.1)   # 绿色，透明度 0.3
        model.attach_to(base)

def merge_pickle(pickle1_path, pickle2_path, output_path=None):
    """
    将 pickle1 中的位姿追加到 pickle2 中，保存到 output_path。
    若 output_path 为 None，则覆盖 pickle2_path。
    """
    with open(pickle1_path, 'rb') as f1:
        data1 = pickle.load(f1)
    with open(pickle2_path, 'rb') as f2:
        data2 = pickle.load(f2)

    # 确保两者都是列表
    if not isinstance(data1, list) or not isinstance(data2, list):
        raise TypeError("Pickle 内容必须为列表")

    merged = data1 + data2   # 将 pickle2 的位姿追加到 pickle1 后面

    save_path = output_path if output_path else pickle2_path
    with open(save_path, 'wb') as f:
        pickle.dump(merged, f)

    print(f"合并完成！共 {len(merged)} 个位姿，保存至 {save_path}")

def merge_multiple_pickle(pickle_path_list, output_path=None):
    """
    合并多个 pickle 文件中的抓取位姿列表。
    参数:
        pickle_path_list: list[str], 待合并的 pickle 文件路径列表
        output_path: str, 可选，合并后的保存路径。若为 None 则保存到列表第一个文件路径
    """
    if not pickle_path_list:
        raise ValueError("pickle_path_list 不能为空")

    merged = []
    for path in pickle_path_list:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise TypeError(f"文件内容不是列表: {path}")
        merged.extend(data)

    save_path = output_path if output_path else pickle_path_list[0]
    with open(save_path, 'wb') as f:
        pickle.dump(merged, f)

    print(f"合并完成！共合并 {len(pickle_path_list)} 个文件，"
          f"总计 {len(merged)} 个位姿，保存至 {save_path}")

if __name__ == "__main__":
    # base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    # mgm.gen_frame().attach_to(base)
    # gripper = dh.Dh76(fingertip_type = "r_76")
    # # 展示pickle文件的抓取姿态
    # show_pickle(base,
    #             pickle_path="result/bottle_dh76_neck_1_2.pickle",
    #             obj_path="../models/bottle.stl",
    #             gripper=gripper,
    #             obj_rgba=[1, 1, 0, 1]
    #             )
    #
    # base.run()

    # # 合并两个pickle文件，把后者放到前者的后面
    # merge_pickle('bottle_dh76_push.pickle', 'bottle_dh76_neck_1_2.pickle', 'result/bottle_dh76_push.pickle')
    # 合并多个pickle文件
    # merge_multiple_pickle(
    #     pickle_path_list=[
    #         'result/bottle_dh76_neck_1.pickle',
    #         'result/bottle_dh76_neck_2.pickle',
    #         'result/bottle_dh76_neck_3.pickle',
    #         'result/bottle_dh76_neck_1_2.pickle',
    #         'result/bottle_dh76_neck_2_2.pickle',
    #     ],
    #     output_path='result/bottle_dh76_neck.pickle'
    # )
    merge_multiple_pickle(
        pickle_path_list=[
            'result/bottle_dh76_front_back.pickle',
            'result/bottle_dh76_head.pickle',
            'result/bottle_dh76_bottom.pickle',
            'result/bottle_dh76_neck.pickle',
            'result/bottle_dh76_handle.pickle',
        ],
        output_path='bottle_dh76_4.pickle'
    )
