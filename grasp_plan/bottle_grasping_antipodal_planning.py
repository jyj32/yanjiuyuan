from wrs import mgm, wd
import math
import numpy as np
import os
import wrs.modeling.collision_model as cm
import wrs.robot_sim.end_effectors.grippers.dh76.dh76 as dh
import wrs.basis.robot_math as rm
import pickle
import tkinter as tk
from tkinter import filedialog
from direct.task import Task

# pickle写入程序
# ---------- 全局变量 ----------
gripper_s = None
gripper_model = None
gripper_local_frame = None # 手爪中心本地坐标系 (RGB)，随夹爪旋转，旋转以此参考
saved_models = []          # 存储每个保存的 ModelCollection 对象
grasp_info_list = []
current_jaw_width = 0.1
current_jaw_center_pos = np.array([0.0, 0.0, 0.025])
current_jaw_center_rotmat = np.eye(3)

base = None  # Panda3D 场景对象

class ControlWindow:
    def __init__(self, root, file_name=None):
        self.root = root
        self.root.title("夹爪位姿控制")
        self.root.geometry("560x640")
        # 允许自由缩放，并设最小尺寸兜底，避免控件被裁掉
        self.root.resizable(True, True)
        self.root.minsize(520, 600)

        self.label_font = ('Arial', 12)
        self.entry_font = ('Arial', 12)
        self.group_font = ('Arial', 12, 'bold')

        self.var = {}
        self.file_name = file_name
        self.create_controls()

        # 让窗口高度按实际内容自适应（至少满足 minsize），避免底部按钮被裁掉
        self.root.update_idletasks()
        _req_h = max(self.root.winfo_reqheight(), 600)
        self.root.geometry(f"560x{_req_h}")

        # 当前夹爪朝向（本地旋转的真实状态），以及上次滑块角度（用于增量式本地旋转）
        # 初始朝向与滑块初值一致：roll=0, pitch=0, yaw=90°（绕世界 Z 转 90°）
        self.grip_rotmat = rm.rotmat_from_euler(0.0, 0.0, math.radians(90.0))
        self._last_angles = {'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0}

    def create_controls(self):
        params = [
            ('group', '位置参数'),
            ('param', 'X 位置 (m)', 'x', -0.1, 0.1, 0.0, 0.001),
            ('param', 'Y 位置 (m)', 'y', -0.1, 0.1, 0.0, 0.001),
            ('param', 'Z 位置 (m)', 'z', 0.0, 0.3, 0.025, 0.001),
            ('group', '绕手爪坐标系角度'),
            ('param', 'Roll (x) (deg)', 'roll', -180, 180, 0.0, 1.0),
            ('param', 'Pitch (y) (deg)', 'pitch', -180, 180, 0.0, 1.0),
            ('param', 'Yaw (z) (deg)', 'yaw', -180, 180, 90.0, 1.0),
            ('group', '手爪宽度'),
            ('param', 'Jaw Width (m)', 'jaw', 0.04, 0.12, 0.1, 0.001),
        ]

        for item in params:
            if item[0] == 'group':
                frame = tk.Frame(self.root)
                frame.pack(pady=(10, 0), fill='x')
                lbl = tk.Label(frame, text=item[1], font=self.group_font, fg='darkblue')
                lbl.pack(anchor='w')
                continue

            _, label, key, from_, to, init, res = item
            frame = tk.Frame(self.root)
            frame.pack(pady=4, fill='x', padx=15)

            lbl = tk.Label(frame, text=label, width=18, anchor='w', font=self.label_font)
            lbl.pack(side=tk.LEFT)

            var = tk.DoubleVar(value=init)
            slider = tk.Scale(frame, from_=from_, to=to, resolution=res,
                              orient=tk.HORIZONTAL, length=220,
                              variable=var,
                              command=lambda v, k=key: self.slider_changed(k))
            slider.pack(side=tk.LEFT, padx=5)

            entry_var = tk.StringVar(value=f"{init:.1f}" if key in ['roll','pitch','yaw'] else f"{init:.3f}")
            entry = tk.Entry(frame, textvariable=entry_var, width=8, font=self.entry_font)
            entry.pack(side=tk.LEFT, padx=5)
            entry.bind('<Return>', lambda e, k=key: self.entry_changed(k))
            entry.bind('<FocusOut>', lambda e, k=key: self.entry_changed(k))

            self.var[key] = {'slider': slider, 'var': var, 'entry_var': entry_var}

        # 按钮框架（第一行）
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=(15, 0))

        tk.Button(btn_frame, text="保存位姿", font=self.label_font,
                  command=self.output_pose).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="保存为", font=self.label_font,
                  command=self.save_as).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="导入", font=self.label_font,
                  command=self.import_poses).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="撤销(u)", font=self.label_font,
                  command=self.undo_last).pack(side=tk.LEFT, padx=5)

        # 按钮框架（第二行）
        btn_frame2 = tk.Frame(self.root)
        btn_frame2.pack(pady=(5, 0))

        tk.Button(btn_frame2, text="重置", font=self.label_font,
                  command=self.reset).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame2, text="隐藏", font=self.label_font,
                  command=self.hide_models).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame2, text="显示", font=self.label_font,
                  command=self.show_models).pack(side=tk.LEFT, padx=5)

        _hint = "就绪（未选择保存路径，请用“保存为”）" if self.file_name is None else f"就绪（保存路径：{os.path.basename(self.file_name)}）"
        self.status_label = tk.Label(self.root, text=_hint, fg="green", font=self.label_font)
        self.status_label.pack(pady=5)

    # ---------- 隐藏/显示（通过 detach/attach） ----------
    def hide_models(self):
        global saved_models
        if not saved_models:
            self.status_label.config(text="没有保存的位姿可隐藏", fg="red")
            return
        for model in saved_models:
            model.detach()          # 从场景移除
        self.status_label.config(text="已隐藏所有保存的位姿", fg="blue")

    def show_models(self):
        global saved_models
        if not saved_models:
            self.status_label.config(text="没有保存的位姿可显示", fg="red")
            return
        for model in saved_models:
            model.attach_to(base)   # 重新添加到场景
        self.status_label.config(text="已显示所有保存的位姿", fg="blue")

    # ---------- 滑块与数值输入回调 ----------
    def slider_changed(self, key):
        val = self.var[key]['var'].get()
        if key in ['roll', 'pitch', 'yaw']:
            self.var[key]['entry_var'].set(f"{val:.1f}")
        else:
            self.var[key]['entry_var'].set(f"{val:.3f}")
        self.update_gripper()

    def entry_changed(self, key):
        try:
            val = float(self.var[key]['entry_var'].get())
            slider = self.var[key]['slider']
            from_, to_ = slider.cget('from'), slider.cget('to')
            if val < from_:
                val = from_
            elif val > to_:
                val = to_
            self.var[key]['var'].set(val)
            if key in ['roll', 'pitch', 'yaw']:
                self.var[key]['entry_var'].set(f"{val:.1f}")
            else:
                self.var[key]['entry_var'].set(f"{val:.3f}")
            self.update_gripper()
        except ValueError:
            val = self.var[key]['var'].get()
            if key in ['roll', 'pitch', 'yaw']:
                self.var[key]['entry_var'].set(f"{val:.1f}")
            else:
                self.var[key]['entry_var'].set(f"{val:.3f}")

    def get_values(self):
        vals = {}
        for key in self.var:
            vals[key] = self.var[key]['var'].get()
        return vals

    # ---------- 更新当前夹爪显示 ----------
    def update_gripper(self, absolute=False):
        global gripper_s, gripper_model, gripper_local_frame
        global current_jaw_width, current_jaw_center_pos, current_jaw_center_rotmat

        vals = self.get_values()
        x, y, z = vals['x'], vals['y'], vals['z']
        roll_rad = math.radians(vals['roll'])
        pitch_rad = math.radians(vals['pitch'])
        yaw_rad = math.radians(vals['yaw'])
        jaw_width = vals['jaw']

        if absolute:
            # 从滑块绝对值重建朝向（用于重置/初始化）：
            # 等价于在本地系下的绝对欧拉角（绕世界固定轴叠加）。
            self.grip_rotmat = (rm.rotmat_from_euler(roll_rad, 0, 0) @
                                rm.rotmat_from_euler(0, pitch_rad, 0) @
                                rm.rotmat_from_euler(0, 0, yaw_rad))
        else:
            # 增量式本地旋转：滑块的“变化量”绕【手爪当前本地坐标系】的对应轴施加
            # （右乘 = 本地系旋转，与 SolidWorks “旋转/平移物体” 的本地旋转一致）。
            droll = vals['roll'] - self._last_angles['roll']
            dpitch = vals['pitch'] - self._last_angles['pitch']
            dyaw = vals['yaw'] - self._last_angles['yaw']
            dR = (rm.rotmat_from_euler(math.radians(droll), 0, 0) @
                  rm.rotmat_from_euler(0, math.radians(dpitch), 0) @
                  rm.rotmat_from_euler(0, 0, math.radians(dyaw)))
            self.grip_rotmat = self.grip_rotmat @ dR

        # 记录本次滑块值，供下次增量计算
        self._last_angles['roll'] = vals['roll']
        self._last_angles['pitch'] = vals['pitch']
        self._last_angles['yaw'] = vals['yaw']

        jaw_center_pos = np.array([x, y, z])

        current_jaw_width = jaw_width
        current_jaw_center_pos = jaw_center_pos
        current_jaw_center_rotmat = self.grip_rotmat

        if gripper_model is not None:
            gripper_model.detach()
        gripper_s.grip_at_by_pose(jaw_center_pos, self.grip_rotmat, jaw_width)
        gripper_model = gripper_s.gen_meshmodel(rgb=[0, 0, 1], alpha=0.8)
        gripper_model.attach_to(base)

        # 手爪中心【本地坐标系】(RGB)：随夹爪一起平移+旋转，旋转以此本地系为参考
        # （与 SolidWorks 旋转/平移物体时物体上的坐标三色轴一致）。
        if gripper_local_frame is not None:
            gripper_local_frame.detach()
        gripper_local_frame = mgm.gen_frame(pos=jaw_center_pos,
                                            rotmat=self.grip_rotmat,
                                            ax_length=0.09,
                                            ax_radius=0.004)
        gripper_local_frame.attach_to(base)

    # ---------- 保存位姿 ----------
    def output_pose(self):
        global grasp_info_list, saved_models
        print("\n===== 当前夹爪位姿 =====")
        print(f"jaw_width  = {current_jaw_width:.4f}")
        print(f"jaw_center_pos = {current_jaw_center_pos.tolist()}")
        print("jaw_center_rotmat =")
        print(current_jaw_center_rotmat)
        print("========================")

        # 保存抓取信息
        hnd_pos = np.array([0, 0, 0])
        hnd_rotmat = np.eye(3)
        grasp_info = [current_jaw_width,
                      current_jaw_center_pos.copy(),
                      current_jaw_center_rotmat.copy(),
                      hnd_pos,
                      hnd_rotmat]
        grasp_info_list.append(grasp_info)

        # 生成保存的位姿模型 (半透明灰色)
        gripper_s.grip_at_by_pose(current_jaw_center_pos, current_jaw_center_rotmat, current_jaw_width)
        save_model = gripper_s.gen_meshmodel(rgb=[0, 0, 0], alpha=0.1)
        save_model.attach_to(base)
        saved_models.append(save_model)   # 存储模型对象

        if self.file_name:
            with open(self.file_name, 'wb') as f:
                pickle.dump(grasp_info_list, f)
            print(f"已保存 {len(grasp_info_list)} 个抓取姿态到 {self.file_name}")
            self.status_label.config(
                text=f"已暂存 {len(grasp_info_list)} 个姿态（→{os.path.basename(self.file_name)}）",
                fg="blue")
        else:
            print(f"已暂存 {len(grasp_info_list)} 个抓取姿态（尚未写文件）")
            self.status_label.config(
                text=f"已暂存 {len(grasp_info_list)} 个姿态（未选路径，请点“保存为”）",
                fg="orange")

    # ---------- 导入已有 pickle 中的抓取姿态 ----------
    def import_poses(self):
        global grasp_info_list, saved_models

        file_path = filedialog.askopenfilename(
            title="选择抓取姿态 pickle 文件",
            filetypes=[("Pickle 文件", "*.pickle"), ("所有文件", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(self.file_name)) if self.file_name else ".",
        )
        if not file_path:
            self.status_label.config(text="未选择文件", fg="red")
            return

        try:
            with open(file_path, 'rb') as f:
                loaded_list = pickle.load(f)
        except Exception as e:
            self.status_label.config(text=f"导入失败: {e}", fg="red")
            return

        if not isinstance(loaded_list, list):
            self.status_label.config(text="文件格式不正确，应为列表", fg="red")
            return

        imported_count = 0
        for grasp_info in loaded_list:
            if len(grasp_info) < 3:
                continue
            jaw_width = grasp_info[0]
            jaw_center_pos = np.asarray(grasp_info[1], dtype=np.float64)
            jaw_center_rotmat = np.asarray(grasp_info[2], dtype=np.float64)

            grasp_info_list.append(grasp_info)

            # 生成半透明灰色模型并显示
            gripper_s.grip_at_by_pose(jaw_center_pos, jaw_center_rotmat, jaw_width)
            save_model = gripper_s.gen_meshmodel(rgb=[0, 0, 0], alpha=0.1)
            save_model.attach_to(base)
            saved_models.append(save_model)
            imported_count += 1

        # 若已选定保存路径，则将合并后的列表写入该文件
        if self.file_name:
            with open(self.file_name, 'wb') as f:
                pickle.dump(grasp_info_list, f)

        self.status_label.config(
            text=f"已导入 {imported_count} 个姿态（共 {len(grasp_info_list)} 个）",
            fg="blue",
        )
        print(f"从 {file_path} 导入了 {imported_count} 个抓取姿态，当前共 {len(grasp_info_list)} 个")

    # ---------- 保存为（选择保存位置与文件名，类似“导入”） ----------
    def save_as(self):
        global grasp_info_list
        file_path = filedialog.asksaveasfilename(
            title="保存抓取姿态为",
            defaultextension=".pickle",
            filetypes=[("Pickle 文件", "*.pickle"), ("所有文件", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(self.file_name)) if self.file_name else ".",
        )
        if not file_path:
            self.status_label.config(text="未选择保存路径", fg="red")
            return

        self.file_name = file_path
        with open(self.file_name, 'wb') as f:
            pickle.dump(grasp_info_list, f)
        self.status_label.config(
            text=f"已保存 {len(grasp_info_list)} 个姿态到 {os.path.basename(self.file_name)}",
            fg="blue",
        )
        print(f"已保存 {len(grasp_info_list)} 个抓取姿态到 {self.file_name}")

    # ---------- 重置 ----------
    def reset(self):
        initials = {'x':0.0, 'y':0.0, 'z':0.025, 'roll':0.0, 'pitch':0.0, 'yaw':90.0, 'jaw':0.1}
        for key, val in initials.items():
            self.var[key]['var'].set(val)
            if key in ['roll', 'pitch', 'yaw']:
                self.var[key]['entry_var'].set(f"{val:.1f}")
            else:
                self.var[key]['entry_var'].set(f"{val:.3f}")
        self.update_gripper(absolute=True)
        self.status_label.config(text="已重置", fg="green")

    # ---------- 撤销 ----------
    def undo_last(self):
        global grasp_info_list, saved_models
        if grasp_info_list and saved_models:
            last_model = saved_models.pop()
            last_model.detach()     # 从场景移除
            grasp_info_list.pop()
            self.status_label.config(
                text=f"已撤销，剩余 {len(grasp_info_list)} 个姿态",
                fg="orange"
            )
        else:
            self.status_label.config(text="没有可撤销的姿态", fg="red")

# ---------- Panda3D 更新任务 ----------
def tk_update_task(task):
    root.update()
    return Task.cont

# ---------- 主程序 ----------
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    # 加载物体 (瓶子)
    this_dir, _ = os.path.split(__file__)
    objpath = os.path.join(this_dir, "..", "models", "bottle.stl")
    if not os.path.exists(objpath):
        objpath = os.path.join(this_dir, "models", "bottle.STL")
        if not os.path.exists(objpath):
            print(f"错误: 找不到物体模型文件")
            exit(1)
    object_tube = cm.CollisionModel(objpath)
    object_tube.rgba = rm.np.array([1, 1, 0, 0.8])
    object_tube.attach_to(base)

    gripper_s = dh.Dh76(fingertip_type = "r_76")

    root.deiconify()
    control = ControlWindow(root)

    # 键盘快捷键
    base.accept('u', control.undo_last)

    base.taskMgr.add(tk_update_task, "tk_update")
    control.update_gripper(absolute=True)
    base.run()