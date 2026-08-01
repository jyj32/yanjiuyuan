import tkinter as tk
from tkinter import messagebox
import pickle
import os
import numpy as np
from wrs import mgm, wd
import wrs.robot_sim.end_effectors.grippers.dh76.dh76 as dh
import wrs.modeling.collision_model as cm
import wrs.basis.robot_math as rm
from direct.task import Task

class ViewerControl:
    def __init__(self, root, base, pickle_path, obj_path=None, obj_rgba=None):
        """
        :param root: tkinter 根窗口
        :param base: wd.World 实例（仿真场景）
        :param pickle_path: 抓取姿态 pickle 文件路径
        :param obj_path: 物体 STL 路径（可选）
        :param obj_rgba: 物体颜色 [R,G,B,A]
        """
        self.root = root
        self.base = base
        self.pickle_path = pickle_path
        self.obj_path = obj_path
        self.obj_rgba = obj_rgba if obj_rgba is not None else [1, 1, 0, 0.5]

        # 加载数据
        self.grasp_list = self._load_pickle()
        self.current_index = 0

        # 初始化夹爪
        self.gripper = dh.Dh76(fingertip_type = "r_76")
        self.current_hand_model = None   # 当前显示的手爪模型

        # 加载物体
        self._load_object()

        # 显示第一个姿态
        self._show_current_grasp()

        # ---------- 创建 tkinter 控制界面 ----------
        self.root.title("抓取姿态查看器")
        self.root.geometry("380x150")

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        btn_prev = tk.Button(btn_frame, text="上一个 (←/P)", command=self.on_previous, width=12)
        btn_prev.pack(side=tk.LEFT, padx=5)

        btn_continue = tk.Button(btn_frame, text="继续 (→/N)", command=self.on_continue, width=12)
        btn_continue.pack(side=tk.LEFT, padx=5)

        btn_delete = tk.Button(btn_frame, text="删除 (Del/X)", command=self.on_delete, width=12)
        btn_delete.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.root, text=self._status_text(), font=("Arial", 12))
        self.status_label.pack(pady=10)

        # ---------- 键盘快捷键 ----------
        # 上一个：Left / p / P
        self.root.bind("<Left>", lambda e: self.on_previous())
        self.root.bind("<Key-p>", lambda e: self.on_previous())
        self.root.bind("<Key-P>", lambda e: self.on_previous())
        # 继续：Right / n / N
        self.root.bind("<Right>", lambda e: self.on_continue())
        self.root.bind("<Key-n>", lambda e: self.on_continue())
        self.root.bind("<Key-N>", lambda e: self.on_continue())
        # 删除：Delete / x / X
        self.root.bind("<Delete>", lambda e: self.on_delete())
        self.root.bind("<Key-x>", lambda e: self.on_delete())
        self.root.bind("<Key-X>", lambda e: self.on_delete())

        # 关闭窗口时关闭仿真
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _load_pickle(self):
        if not os.path.exists(self.pickle_path):
            raise FileNotFoundError(f"找不到文件: {self.pickle_path}")
        with open(self.pickle_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise TypeError("Pickle 内容必须为列表")
        return data

    def _load_object(self):
        if self.obj_path and os.path.exists(self.obj_path):
            obj_model = cm.CollisionModel(self.obj_path)
            obj_model.rgba = np.array(self.obj_rgba)
            obj_model.attach_to(self.base)

    def _show_current_grasp(self):
        # 清除旧手爪
        if self.current_hand_model is not None:
            self.current_hand_model.detach()
            self.current_hand_model = None

        if not self.grasp_list:
            return

        if self.current_index >= len(self.grasp_list):
            self.current_index = len(self.grasp_list) - 1

        grasp_info = self.grasp_list[self.current_index]
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        self.gripper.grip_at_by_pose(jaw_center_pos, jaw_center_rotmat, jaw_width)
        self.current_hand_model = self.gripper.gen_meshmodel(rgb=[0, 1, 0], alpha=0.3)
        self.current_hand_model.attach_to(self.base)

        # Panda3D 会在下一帧自动刷新，无需额外调用

    def _status_text(self):
        total = len(self.grasp_list)
        if total == 0:
            return "无姿态数据"
        return f"姿态 {self.current_index+1} / {total}"

    def on_continue(self):
        if self.current_index < len(self.grasp_list) - 1:
            self.current_index += 1
            self._show_current_grasp()
            self.status_label.config(text=self._status_text())
        else:
            messagebox.showinfo("提示", "已是最后一个姿态")

    def on_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._show_current_grasp()
            self.status_label.config(text=self._status_text())
        else:
            messagebox.showinfo("提示", "已是第一个姿态")

    def on_delete(self):
        if not self.grasp_list:
            messagebox.showwarning("提示", "没有姿态可删除")
            return
        # 删除当前姿态
        del self.grasp_list[self.current_index]
        # 调整索引
        if self.current_index >= len(self.grasp_list):
            self.current_index = max(0, len(self.grasp_list) - 1)
        # 保存到文件
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self.grasp_list, f)
        # 更新显示
        self._show_current_grasp()
        self.status_label.config(text=self._status_text())
        messagebox.showinfo("完成", "已删除当前姿态并保存")

    def on_close(self):
        # 安全关闭仿真窗口
        if hasattr(self.base, 'close'):
            self.base.close()
        elif hasattr(self.base, 'destroy'):
            self.base.destroy()
        self.root.destroy()


# ---------- 主程序 ----------
if __name__ == '__main__':
    # 创建 tkinter 根窗口（先隐藏）
    root = tk.Tk()
    root.withdraw()

    # 创建仿真世界（独立窗口，默认支持鼠标交互）
    base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0, 0.5])
    mgm.gen_frame().attach_to(base)

    # 创建控制窗口，传入必要的参数
    viewer = ViewerControl(
        root=root,
        base=base,
        pickle_path="./result/bottle_dh76_front_back.pickle",   # 修改为您的 pickle 路径
        obj_path="../models/bottle.stl",    # 物体模型路径（可选）
        obj_rgba=[1, 1, 0, 0.6]             # 黄色半透明
    )

    # 显示 tk 窗口
    root.deiconify()

    # 定义 Panda3D 任务：持续更新 tkinter 事件循环（闭包捕获 root）
    def tk_update_task(task):
        root.update()
        return Task.cont

    # 将任务加入 Panda3D 任务管理器
    base.taskMgr.add(tk_update_task, "tk_update")

    # 启动 Panda3D 主循环
    base.run()