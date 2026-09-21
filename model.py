import math
from abc import ABC, abstractmethod
import torch.nn.functional as F
from typing import Any
from torch.func import vmap
import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn
from torch.distributions import MultivariateNormal, Normal
from torch.nn.functional import batch_norm, sigmoid


class MZIMatrix(ABC):
    @abstractmethod
    def forward(self, x):
        ...

    @abstractmethod
    def update_voltage(self, v_diff):
        ...

    @abstractmethod
    def hardware_miss_alignment(self, eps):
        ...

    @abstractmethod
    def temperature_shift(self, T):
        ...

    @abstractmethod
    def power_loss(self, power_loss):
        ...


class OptimizerAgent(ABC):
    @abstractmethod
    def action(self, state):
        """state: x,y_theta"""
        ...


def symmetric_mzi_matrix(phs_layer):
    assert phs_layer.shape[0] % 2 == 0, "phs_layer must be even"
    device = phs_layer.device  # 【关键1】：从输入的 phase layer 推断 device，不写死 cuda

    matrices = []
    # 【优化】：预先计算好常数矩阵，避免在循环中重复创建
    base_mat = (math.sqrt(2) / 2) * torch.tensor([[1, 1j], [1j, 1]], dtype=torch.complex64, device=device)

    for i in range(phs_layer.shape[0] // 2):
        mat3 = torch.diag(torch.exp(1j * phs_layer[2 * i: 2 * i + 2]))
        matrix = base_mat @ mat3
        matrices.append(matrix)

    # 【优化】：将列表解包一次性构建块对角矩阵，比循环里拼接更快
    matrix_layer = torch.block_diag(*matrices)
    return matrix_layer


def mzi_mesh(global_phi, parallel=8):
    device = global_phi.device  # 【关键1】
    matrix_mesh = torch.eye(parallel, dtype=torch.complex64, device=device)

    for lay_idx in range(global_phi.shape[0]):
        layer_phi = global_phi[lay_idx]
        if lay_idx in [2, 4, 6, 8]:
            matrix_layer = symmetric_mzi_matrix(layer_phi[1:-1])
            # padding 两端用 [1]
            padding = torch.ones((1,), dtype=torch.complex64, device=device)
            matrix_layer = torch.block_diag(padding, matrix_layer, padding)
            matrix_mesh = matrix_mesh @ matrix_layer
        else:
            matrix_layer = symmetric_mzi_matrix(layer_phi)
            matrix_mesh = matrix_mesh @ matrix_layer
    return matrix_mesh


class SimMZIMitrix(torch.nn.Module, MZIMatrix):
    def __init__(self, layer_num, parallel):
        super().__init__()
        self.layer_num = layer_num
        self.parallel = parallel
        parameter_shape = (layer_num, parallel)
        num_phase_shifters = math.prod(parameter_shape)

        # 电压到相位的标定参数；默认保持原有的 phi = V + c 模型。
        self.register_buffer("shifter_a", torch.zeros(parameter_shape, dtype=torch.float32))
        self.register_buffer("shifter_b", torch.ones(parameter_shape, dtype=torch.float32))
        self.register_buffer("shifter_c", torch.randn(parameter_shape, dtype=torch.float32))
        self.register_buffer("voltage", torch.zeros(parameter_shape, dtype=torch.float32))

        # 默认值不引入新的非理想性，以保持原有仿真行为。
        self.register_buffer("fabrication_phase_error", torch.zeros(parameter_shape, dtype=torch.float32))
        self.register_buffer("heater_resistance", torch.ones(parameter_shape, dtype=torch.float32))
        self.register_buffer("crosstalk_matrix", torch.zeros((num_phase_shifters, num_phase_shifters), dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(25.0, dtype=torch.float32))
        self.register_buffer("reference_temperature", torch.tensor(25.0, dtype=torch.float32))
        self.register_buffer("temperature_coefficient", torch.zeros(parameter_shape, dtype=torch.float32))
        self.register_buffer("drift_phase", torch.zeros(parameter_shape, dtype=torch.float32))
        self.register_buffer("drift_std", torch.zeros(parameter_shape, dtype=torch.float32))
        self.register_buffer("drift_time_constant", torch.tensor(60.0, dtype=torch.float32))
        self.register_buffer("power_transmission", torch.tensor(1.0, dtype=torch.float32))

        self.register_buffer("global_phi", torch.zeros(parameter_shape, dtype=torch.float32))
        self.mesh_matrix = torch.eye(parallel, dtype=torch.complex64)
        self.nonlinear = nn.Identity()# F.sigmoid
        self.update_voltage(torch.zeros(parameter_shape, dtype=torch.float32), dt=0.0)

    def step(self, x):
        return self.forward(x)

    def forward(self, x):
        """get amp and phase result"""
        x = x.to(dtype=torch.complex64)

        # 确保计算图中的 mesh_matrix 和输入 x 在同一个 device 上
        if self.mesh_matrix.device != x.device:
            self.mesh_matrix = self.mesh_matrix.to(x.device)

        detected = (x @ self.mesh_matrix).abs().square().float()
        return self.nonlinear(detected)

    @torch.no_grad()
    def _evolve_drift(self, dt):
        """按 Ornstein-Uhlenbeck 过程推进慢相位漂移。"""
        dt = float(dt)
        if dt < 0:
            raise ValueError("dt 必须非负")
        if dt == 0:
            return
        tau = self.drift_time_constant.clamp_min(torch.finfo(torch.float32).eps)
        rho = torch.exp(-torch.as_tensor(dt, device=self.global_phi.device) / tau)
        innovation = self.drift_std * torch.sqrt((1 - rho.square()).clamp_min(0))
        self.drift_phase.mul_(rho).add_(innovation * torch.randn_like(self.drift_phase))

    @torch.no_grad()
    def _refresh_physics(self):
        power = self.voltage.square() / self.heater_resistance.clamp_min(torch.finfo(torch.float32).eps)
        phase_crosstalk = (power.reshape(-1) @ self.crosstalk_matrix.T).view_as(power)
        phase_temperature = self.temperature_coefficient * (self.temperature - self.reference_temperature)
        phase = (self.shifter_a * self.voltage.square() + self.shifter_b * self.voltage + self.shifter_c +
                 self.fabrication_phase_error + phase_crosstalk + phase_temperature + self.drift_phase)
        self.global_phi.copy_(phase)
        self.mesh_matrix = mzi_mesh(self.global_phi) * self.power_transmission.sqrt()

    @torch.no_grad()
    def update_voltage(self, v_diff, dt=1.0):
        voltage = v_diff.to(self.global_phi.device, dtype=self.global_phi.dtype).view(self.layer_num, self.parallel)
        self.voltage.copy_(voltage)
        self._evolve_drift(dt)
        self._refresh_physics()

    @torch.no_grad()
    def hardware_miss_alignment(self, eps):
        """生成固定制造失准相位，eps 的单位为弧度标准差。"""
        eps = torch.as_tensor(eps, dtype=self.global_phi.dtype, device=self.global_phi.device)
        self.fabrication_phase_error.copy_(torch.randn_like(self.fabrication_phase_error) * eps)
        self._refresh_physics()

    @torch.no_grad()
    def set_crosstalk_matrix(self, crosstalk_matrix):
        """设置 C[受扰通道, 加热通道]，单位为弧度每功率单位。"""
        matrix = torch.as_tensor(
            crosstalk_matrix, dtype=self.crosstalk_matrix.dtype, device=self.crosstalk_matrix.device
        )
        if matrix.shape != self.crosstalk_matrix.shape:
            raise ValueError(f"crosstalk_matrix 的形状必须为 {tuple(self.crosstalk_matrix.shape)}")
        self.crosstalk_matrix.copy_(matrix)
        self._refresh_physics()

    @torch.no_grad()
    def set_temperature_coefficient(self, coefficient):
        """设置温度相移系数，可传入标量或每个相移器的系数。"""
        coefficient = torch.as_tensor(
            coefficient, dtype=self.temperature_coefficient.dtype, device=self.temperature_coefficient.device
        )
        try:
            coefficient = torch.broadcast_to(coefficient, self.temperature_coefficient.shape)
        except RuntimeError as exc:
            raise ValueError(f"coefficient 必须可广播到 {tuple(self.temperature_coefficient.shape)}") from exc
        self.temperature_coefficient.copy_(coefficient)
        self._refresh_physics()

    @torch.no_grad()
    def configure_drift(self, std, time_constant=None):
        """设置 OU 漂移的稳态标准差及可选相关时间。"""
        std = torch.as_tensor(std, dtype=self.drift_std.dtype, device=self.drift_std.device)
        try:
            std = torch.broadcast_to(std, self.drift_std.shape)
        except RuntimeError as exc:
            raise ValueError(f"std 必须可广播到 {tuple(self.drift_std.shape)}") from exc
        if bool(torch.any(std < 0).item()):
            raise ValueError("漂移标准差必须非负")
        self.drift_std.copy_(std)
        if time_constant is not None:
            time_constant = torch.as_tensor(
                time_constant, dtype=self.drift_time_constant.dtype, device=self.drift_time_constant.device
            )
            if time_constant.numel() != 1 or bool((time_constant <= 0).item()):
                raise ValueError("漂移相关时间必须为正标量")
            self.drift_time_constant.copy_(time_constant)

    @torch.no_grad()
    def temperature_shift(self, T):
        """设置封装绝对温度，单位须与参考温度一致。"""
        self.temperature.copy_(torch.as_tensor(T, dtype=self.temperature.dtype, device=self.temperature.device))
        self._refresh_physics()

    @torch.no_grad()
    def power_loss(self, power_loss):
        """设置总功率损耗比例；0.2 表示输出功率损失 20%。"""
        loss = torch.as_tensor(power_loss, dtype=self.power_transmission.dtype, device=self.power_transmission.device)
        if loss.numel() != 1 or bool(torch.any((loss < 0) | (loss >= 1)).item()):
            raise ValueError("power_loss 必须是区间 [0, 1) 内的标量")
        self.power_transmission.copy_(1 - loss)
        self._refresh_physics()

    @torch.no_grad()
    def advance_time(self, dt):
        """在电压不变时仅推进慢漂移。"""
        self._evolve_drift(dt)
        self._refresh_physics()
# class RLEnv:
#     def __init__(self):
#         ...
#     def step(self,act,x,y_target):
#         self.mzi.update_voltage(act)
#         y_theta=self.mzi.forward(x)
#         obs=y_theta
#         return obs
class TiledMZIMatrix(nn.Module, MZIMatrix):
    def __init__(self, in_features, out_features, layer_num, parallel=8, device="cuda:0"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_num = layer_num
        self.parallel = parallel

        self.in_blocks = math.ceil(in_features / parallel)
        self.out_blocks = math.ceil(out_features / parallel)

        self.pad_in = self.in_blocks * parallel - in_features
        self.pad_out = self.out_blocks * parallel - out_features

        param_shape = (self.in_blocks, self.out_blocks, layer_num, parallel)
        mzi_property_shape = (layer_num, parallel)
        num_phase_shifters_per_tile = layer_num * parallel

        # 1. 注册基础物理参数 buffer
        self.register_buffer("global_phi", torch.randn(param_shape, dtype=torch.float32))
        self.register_buffer("shifter_a", torch.zeros(mzi_property_shape, dtype=torch.float32))
        self.register_buffer("shifter_b", torch.ones(mzi_property_shape, dtype=torch.float32))
        self.register_buffer("shifter_c", torch.randn(mzi_property_shape, dtype=torch.float32))
        self.register_buffer("voltage", torch.zeros(param_shape, dtype=torch.float32))

        # 各非理想性均作为 buffer 保存，默认值不改变原始理想仿真。
        # 串扰仅在单个 8x8 tile 内建模，避免无物理依据的全局稠密矩阵。
        self.register_buffer("fabrication_phase_error", torch.zeros(param_shape, dtype=torch.float32))
        self.register_buffer("heater_resistance", torch.ones(param_shape, dtype=torch.float32))
        self.register_buffer(
            "crosstalk_matrix",
            torch.zeros((num_phase_shifters_per_tile, num_phase_shifters_per_tile), dtype=torch.float32),
        )
        self.register_buffer("temperature", torch.tensor(25.0, dtype=torch.float32))
        self.register_buffer("reference_temperature", torch.tensor(25.0, dtype=torch.float32))
        self.register_buffer("temperature_coefficient", torch.zeros(param_shape, dtype=torch.float32))
        self.register_buffer("drift_phase", torch.zeros(param_shape, dtype=torch.float32))
        self.register_buffer("drift_std", torch.zeros(param_shape, dtype=torch.float32))
        self.register_buffer("drift_time_constant", torch.tensor(60.0, dtype=torch.float32))
        self.register_buffer("power_transmission", torch.ones((self.in_blocks, self.out_blocks), dtype=torch.float32))

        # 【修复1】：将 mesh_matrices 也注册为 buffer！让它归 PyTorch 管。
        mesh_shape = (self.in_blocks, self.out_blocks, self.parallel, self.parallel)
        self.register_buffer("mesh_matrices", torch.zeros(mesh_shape, dtype=torch.complex64))

        self.nonlinear = lambda x: x #sigmoid

        # 初始化动作 (注意：此时依然在 CPU 上，等会会被 .to 转移)
        initial_action = torch.zeros(math.prod(param_shape), dtype=torch.float32)
        self.update_voltage(initial_action, dt=0.0)



    @torch.no_grad()
    def _evolve_drift(self, dt):
        """按 Ornstein-Uhlenbeck 过程推进慢相位漂移。"""
        dt = float(dt)
        if dt < 0:
            raise ValueError("dt 必须非负")
        if dt == 0:
            return
        tau = self.drift_time_constant.clamp_min(torch.finfo(torch.float32).eps)
        rho = torch.exp(-torch.as_tensor(dt, device=self.global_phi.device) / tau)
        innovation = self.drift_std * torch.sqrt((1 - rho.square()).clamp_min(0))
        self.drift_phase.mul_(rho).add_(innovation * torch.randn_like(self.drift_phase))

    @torch.no_grad()
    def _refresh_physics(self):
        """根据电压与全部物理状态重建 MZI 网格。"""
        power = self.voltage.square() / self.heater_resistance.clamp_min(torch.finfo(torch.float32).eps)
        power_flat = power.view(-1, self.layer_num * self.parallel)
        phase_crosstalk = (power_flat @ self.crosstalk_matrix.T).view_as(power)
        phase_temperature = self.temperature_coefficient * (self.temperature - self.reference_temperature)
        phase = (self.shifter_a * self.voltage.square() + self.shifter_b * self.voltage + self.shifter_c +
                 self.fabrication_phase_error + phase_crosstalk + phase_temperature + self.drift_phase)
        self.global_phi.copy_(phase)

        phi_flat = self.global_phi.view(self.in_blocks * self.out_blocks, self.layer_num, self.parallel)
        matrices_flat = vmap(mzi_mesh, in_dims=0)(phi_flat)
        matrices = matrices_flat.view(self.in_blocks, self.out_blocks, self.parallel, self.parallel)
        amplitude_transmission = self.power_transmission.sqrt().unsqueeze(-1).unsqueeze(-1)
        self.mesh_matrices.copy_(matrices * amplitude_transmission)

    @torch.no_grad()
    def update_voltage(self, v_diff, dt=1.0):
        voltage = v_diff.to(self.global_phi.device, dtype=self.global_phi.dtype)
        voltage = voltage.view(self.in_blocks, self.out_blocks, self.layer_num, self.parallel)
        self.voltage.copy_(voltage)
        self._evolve_drift(dt)
        self._refresh_physics()

    @torch.no_grad()
    def hardware_miss_alignment(self, eps):
        """生成固定制造失准相位，eps 的单位为弧度标准差。"""
        eps = torch.as_tensor(eps, dtype=self.global_phi.dtype, device=self.global_phi.device)
        self.fabrication_phase_error.copy_(torch.randn_like(self.fabrication_phase_error) * eps)
        self._refresh_physics()

    @torch.no_grad()
    def set_crosstalk_matrix(self, crosstalk_matrix):
        """设置 tile 内 C[受扰通道, 加热通道]，单位为弧度每功率单位。"""
        matrix = torch.as_tensor(
            crosstalk_matrix, dtype=self.crosstalk_matrix.dtype, device=self.crosstalk_matrix.device
        )
        if matrix.shape != self.crosstalk_matrix.shape:
            raise ValueError(f"crosstalk_matrix 的形状必须为 {tuple(self.crosstalk_matrix.shape)}")
        self.crosstalk_matrix.copy_(matrix)
        self._refresh_physics()

    @torch.no_grad()
    def set_temperature_coefficient(self, coefficient):
        """设置温度相移系数，可传入标量或每个相移器的系数。"""
        coefficient = torch.as_tensor(
            coefficient, dtype=self.temperature_coefficient.dtype, device=self.temperature_coefficient.device
        )
        try:
            coefficient = torch.broadcast_to(coefficient, self.temperature_coefficient.shape)
        except RuntimeError as exc:
            raise ValueError(f"coefficient 必须可广播到 {tuple(self.temperature_coefficient.shape)}") from exc
        self.temperature_coefficient.copy_(coefficient)
        self._refresh_physics()

    @torch.no_grad()
    def configure_drift(self, std, time_constant=None):
        """设置 OU 漂移的稳态标准差及可选相关时间。"""
        std = torch.as_tensor(std, dtype=self.drift_std.dtype, device=self.drift_std.device)
        try:
            std = torch.broadcast_to(std, self.drift_std.shape)
        except RuntimeError as exc:
            raise ValueError(f"std 必须可广播到 {tuple(self.drift_std.shape)}") from exc
        if bool(torch.any(std < 0).item()):
            raise ValueError("漂移标准差必须非负")
        self.drift_std.copy_(std)
        if time_constant is not None:
            time_constant = torch.as_tensor(
                time_constant, dtype=self.drift_time_constant.dtype, device=self.drift_time_constant.device
            )
            if time_constant.numel() != 1 or bool((time_constant <= 0).item()):
                raise ValueError("漂移相关时间必须为正标量")
            self.drift_time_constant.copy_(time_constant)

    @torch.no_grad()
    def temperature_shift(self, T):
        """设置封装绝对温度，单位须与参考温度一致。"""
        self.temperature.copy_(torch.as_tensor(T, dtype=self.temperature.dtype, device=self.temperature.device))
        self._refresh_physics()

    @torch.no_grad()
    def power_loss(self, power_loss):
        """设置每个 tile 的功率损耗比例；标量会自动广播。"""
        loss = torch.as_tensor(power_loss, dtype=self.power_transmission.dtype, device=self.power_transmission.device)
        if bool(torch.any((loss < 0) | (loss >= 1)).item()):
            raise ValueError("power_loss 元素必须位于区间 [0, 1)")
        try:
            loss = torch.broadcast_to(loss, self.power_transmission.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"power_loss 必须为标量或可广播到 {tuple(self.power_transmission.shape)}"
            ) from exc
        self.power_transmission.copy_(1 - loss)
        self._refresh_physics()

    @torch.no_grad()
    def advance_time(self, dt):
        """在电压不变时仅推进慢漂移。"""
        self._evolve_drift(dt)
        self._refresh_physics()

    def forward(self, x):
        batch_size = x.size(0)

        if self.pad_in > 0:
            x = F.pad(x, (0, self.pad_in))

        x = x.to(dtype=torch.complex64)
        x_blocks = x.view(batch_size, self.in_blocks, self.parallel)

        # 【核心优化】：使用 einsum 替代双重 for 循环！
        # b: batch, i: in_blocks, p: parallel (input)
        # j: out_blocks, q: parallel (output)
        # 计算结果 out_complex 的形状为: [batch_size, in_blocks, out_blocks, parallel]
        out_complex = torch.einsum('bip, ijpq -> bijq', x_blocks, self.mesh_matrices)

        # 分别求实部和虚部的平方和，得到光强
        out_mag = out_complex.real.square() + out_complex.imag.square()

        # 在 in_blocks 维度 (dim=1) 上进行非相干累加
        # 结果形状变为: [batch_size, out_blocks, parallel]
        out_blocks = out_mag.sum(dim=1)

        out_flat = out_blocks.view(batch_size, -1)

        if self.pad_out > 0:
            out_flat = out_flat[:, :self.out_features]

        return self.nonlinear(out_flat.float())

    def step(self, x):
        return self.forward(x)

class DNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.model(x)


import torch
import torch.nn as nn
from lightning import LightningModule
from torch.distributions import MultivariateNormal


class PPOFixedMZI(LightningModule):
    def __init__(self, M_samples, act_space, actor_lr, mzi, epsilon=0.2, n_updates_per_batch=5):
        super().__init__()
        self.automatic_optimization = False
        self.env = mzi

        self.M_samples = M_samples  # 对应论文中的 M：每次采样多少组不同的 MZI 参数去试错
        self.act_space = act_space
        self.epsilon = epsilon
        self.actor_lr = actor_lr
        self.n_updates_per_batch = n_updates_per_batch

        # 【关键改变1】：Actor 不再是神经网络，而是直接的物理参数分布的均值 (mu)
        # 初始化为你猜测的一组电压，或者全 0
        self.mu = nn.Parameter(torch.zeros(act_space, dtype=torch.float32))

        # 【核心优化】：不再注册全零的协方差矩阵矩阵！
        # 论文方差是 0.04，标准差就是 sqrt(0.04) = 0.2
        # 我们只保留一个一维的标准差向量（内存占用从 1.5TB 降到几 KB）
        self.register_buffer('sigma', torch.full((act_space,), 0.2, dtype=torch.float32))

        self.loss_fn = torch.nn.functional.cross_entropy

    def configure_optimizers(self):
        # 优化器现在只优化 mu 这一个张量
        actor_optim = torch.optim.Adam([self.mu], lr=self.actor_lr)
        return actor_optim

    def get_action_and_logprob(self, action=None):
        # 【核心优化】：使用标准 Normal 分布代替复杂的 MultivariateNormal
        dist = Normal(self.mu, self.sigma)

        if action is None:
            action = dist.sample()

        # 【核心点】：各个物理参数是独立的，多元正态分布的对数概率
        # 等于每个独立一元正态分布对数概率的总和。
        # 如果 action 是单样本 [act_space]，则 sum(-1) 变成标量
        # 如果 action 是批次 [M_samples, act_space]，则 sum(-1) 变成 [M_samples]
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    @torch.no_grad()
    def _apply_policy_mean_to_env(self):
        """Program the MZI with the deterministic PPO policy used for evaluation."""
        self.env.update_voltage(self.mu.detach())

    def on_validation_epoch_start(self):
        # Evaluate the learned policy mean, not the final random rollout action.
        self._apply_policy_mean_to_env()

    def on_test_epoch_start(self):
        # The held-out test set must use the same deterministic policy.
        self._apply_policy_mean_to_env()

    def _evaluate_batch(self, batch_data, metric_name):
        x_batch, y_target_batch = batch_data
        x_batch,y_target_batch=x_batch.cuda(),y_target_batch.cuda()
        x_batch = x_batch.view(x_batch.size(0), -1)
        y_theta_batch = self.env.step(x_batch)
        y_theta_batch=torch.argmax(y_theta_batch, dim=1)
        batch_acc=torch.sum(y_theta_batch==y_target_batch)/y_target_batch.size(0)
        self.log(
            metric_name,
            batch_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=x_batch.size(0),
        )

    def validation_step(self, batch_data, batch_idx):
        self._evaluate_batch(batch_data, "val_acc")

    def test_step(self, batch_data, batch_idx):
        self._evaluate_batch(batch_data, "test_acc")
    def training_step(self, batch_data):
        # batch_data: 包含了一批输入信号 X 和对应的 Y_target [batch_size, parallel]
        x_batch, y_target_batch = batch_data
        x_batch,y_target_batch=x_batch.cuda(),y_target_batch.cuda()
        # 【关键修复】：将图像展平为 [batch_size, 784]
        x_batch = x_batch.view(x_batch.size(0), -1)
        actor_optim = self.optimizers()

        # 1. 采样 M 组不同的 MZI 动作 (对应论文 Step 1)
        # shape: [M_samples, act_space]
        sampled_actions = torch.zeros((self.M_samples, self.act_space), device=self.device)
        old_log_probs = torch.zeros(self.M_samples, device=self.device)
        rewards = torch.zeros(self.M_samples, device=self.device)

        with torch.no_grad():
            for i in range(self.M_samples):
                action, log_prob = self.get_action_and_logprob()
                sampled_actions[i] = action
                old_log_probs[i] = log_prob

                # 2. 物理评估 (对应论文 Step 2)
                # 将采样到的这组 action（电压）写入 MZI
                self.env.update_voltage(action)

                # 用这*一组*物理参数，跑完*一整批*的数据 X，得到输出
                y_theta_batch = self.env.step(x_batch)
                # y_theta_batch = torch.argmax(y_theta_batch, dim=1)

                # 3. 计算这一组参数在这批数据上的总表现 (对应论文 Step 3)
                # reward 是一个标量
                reward = -self.loss_fn(y_theta_batch, y_target_batch)
                rewards[i] = reward

            # 计算优势函数 Advantage
            # 因为没有序列决策（没有时间步），所以不需要计算 rwtg 和 Critic 网络，直接用 Reward 标准化即可
            # A_k = (R - R_mean) / (R_std + 1e-10)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

        # 4. PPO 数字策略更新 (对应论文 Step 4)
        for _ in range(self.n_updates_per_batch):
            # 重新计算当前 mu 下，那些 sampled_actions 的概率
            _, curr_log_probs = self.get_action_and_logprob(sampled_actions)

            # ✅ 工业级安全的防爆写法
            log_ratio = curr_log_probs - old_log_probs

            # 将对数差值限制在 [-20, 20] 之间。
            # torch.exp(20) 约为 4.8x10^8，足够策略进行剧烈优化，但绝对不会变成 inf 导致硬件溢出
            ratios = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages

            # 我们要最大化 reward，所以 loss 加负号
            actor_loss = -torch.min(surr1, surr2).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

        self.log("train_actor_loss", actor_loss.item(), prog_bar=True)
        self.log("batch_mean_reward", rewards.mean().item(), prog_bar=True)


# 模型1：你的方案（8x8 + ReLU）
class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)  # 8x8矩阵
        self.activation = nn.ReLU()
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    def validation_step(self, batch_data):
        x_batch, y_target_batch = batch_data
        y_theta_batch = self.forward(x_batch)
        y_theta_batch = torch.argmax(y_theta_batch, dim=1)
        batch_acc = torch.sum(y_theta_batch==y_target_batch)/y_target_batch.size(0)
        self.log("val_acc", batch_acc.item(), on_step=False, on_epoch=True, prog_bar=True)


    def training_step(self, batch_data):
        x_batch, y_target_batch = batch_data
        y_theta_batch = self.forward(x_batch)
        loss = torch.nn.functional.cross_entropy(y_theta_batch, y_target_batch)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def forward(self, x):
        return self.activation(self.linear(x))


if __name__ == '__main__':
    symmetric_mzi_matrix(np.ones((6,)))
    matrix_mesh = mzi_mesh(np.ones((10, 8)))
    print(matrix_mesh.shape)
#& 'D:\PycharmProjects\pythonProject\.venv\Scripts\mlflow.exe' server --backend-store-uri 'sqlite:///D:/TrainONN-RL-w-o-NN/TrainONN-RL-w-o-NN/mlflow.db' --port 5000
