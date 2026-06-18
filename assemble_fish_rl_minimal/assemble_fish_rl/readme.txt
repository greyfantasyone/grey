方案 A：只用尾部 4 个关节，同时完成 fish base 的稳定与尾端末端执行器 ee 的目标定位/保持。
- 鱼体 base 是 free-floating（自由漂浮）
- 智能体不能控制独立推进器
- 只能控制尾部 4 个关节
- 任务要求同时做到：
  1. 尾端 ee 靠近目标 target_site
  2. 让 base 不要漂得太远
  3. 在目标附近保持稳定一段时间，才算成功

这类任务本质上是一个强耦合、延迟效应明显、部分时序依赖很强的控制问题，所以工程中默认使用：
- RecurrentPPO（LSTM 版 PPO）
- VecNormalize
- 课程学习
- 扰动随机化
- 可选“弱辅助稳定器”


1. phy3.02_schemeA_grasp.xml

这是默认训练使用的任务版 XML。


2. phy3.02_schemeA_grasp_ellipsoid.xml

这是增强耦合版 XML。

作用：
- 在尾部 4 段增加隐藏的 ellipsoid fluid geoms
- 使尾部动作与水动力耦合更强
- 更容易让“尾部动作影响漂浮 base”的现象出现


3. fish_floating_grasp_env.py
定义强化学习环境。


4.1关键说明：
- --xml：训练用模型文件
- --timesteps：总训练步数
- --n-envs：并行环境数
- --save：模型输出路径前缀
- --algo：recurrentppo 或 ppo
- --resume：恢复已有模型
- --vecnorm：恢复已有归一化器
- --assist-mode：辅助模式
- --curriculum-max-target-scale：目标课程范围缩放
- --learning-rate：学习率
- --n-steps：每个 rollout 的长度
- --batch-size：训练 batch 大小
- --n-epochs：每轮 PPO 更新次数

4.2 默认算法设置

如果 --algo recurrentppo，则使用：
- MlpLstmPolicy
- lstm_hidden_size=256
- n_lstm_layers=1
- shared_lstm=False
- enable_critic_lstm=True

普通 PPO 则使用 MlpPolicy。

默认网络：
- net_arch=dict(pi=[256, 256], vf=[256, 256])

说明策略网络和值函数网络都是 2 层 256。

4.3 默认超参数

RecurrentPPO 默认：
- learning_rate = 3e-4（线性衰减）
- n_steps = 512
- batch_size = 256
- n_epochs = 10
- gamma = 0.99
- gae_lambda = 0.95
- clip_range = 0.2（线性衰减）
- ent_coef = 0.002
- vf_coef = 0.5
- max_grad_norm = 0.5

这些参数决定训练稳定性和收敛速度。



5. play_recurrent_ppo_floating_grasp.py

回放脚本

6. inspect_tail_coupling.py

物理合理性验证脚本。

- 不训练
- 直接给尾部关节施加开环周期动作
- 观察 base 是否发生漂移

输出包括：
- start_base
- end_base
- net_drift
- range_xyz

如果这些几乎为 0，说明：
尾部动作几乎没有对 free-floating base 产生有效耦合

那纯方案 A 就很难训出来。

7. make_schemeA_grasp_xml.py

这是 XML 生成/补丁脚本。

作用：
- 从 phy3.02_with_ee.xml 生成任务版 XML
- 自动添加目标与相机
- 给部分 joint 补 damping
- 可选给尾部加 ellipsoid fluid geom

7.1 主要功能点

patch_joint_damping()
给以下关节补阻尼：
- Joint_1 -> 0.20
- Joint_2 -> 0.30
- Joint_3 -> 0.30
- Joint_4 -> 0.25

作用：
- 增强数值稳定性
- 避免尾部关节过于抖动

add_task_objects()
会增加：
- head_proxy
- target_body
- target_geom
- target_site
- 相机 grasp_side
- 相机 grasp_top

add_tail_ellipsoid_fluid_geoms()
给 Link_1~4 增加隐藏流体耦合几何体。

默认参数：
- fitscale = 0.90
- fluidcoef = "0.50 0.15 1.20 0.10 0.10"

执行顺序：

python -m pip install -r .\requirements.txt
验证尾部动作是否能影响漂浮 base
python .\inspect_tail_coupling.py --xml .\phy3.02_schemeA_grasp.xml
结果判断：
重点看终端输出中的：
- net_drift
- range_xyz
只要不是几乎 0，就说明存在有效耦合。
做环境自检：
python .\smoke_test.py --xml .\phy3.02_schemeA_grasp.xml
成功时应看到类似：
[OK] smoke test finished
纯方案 A 训练：
python .\train_recurrent_ppo_floating_grasp.py `
  --xml .\phy3.02_schemeA_grasp.xml `
  --algo recurrentppo `
  --assist-mode off `
  --timesteps 1500000 `
  --n-envs 4 `
  --save .\checkpoints\floating_grasp_pureA

- 这是纯尾部控制，不使用辅助器
- 最符合方案 A 定义
- 但训练难度最高

纯方案 A 难收敛，改用衰减辅助训练：
python .\train_recurrent_ppo_floating_grasp.py `
  --xml .\phy3.02_schemeA_grasp.xml `
  --algo recurrentppo `
  --assist-mode decay `
  --assist-max-strength 0.35 `
  --assist-decay-end 0.35 `
  --timesteps 1200000 `
  --n-envs 4 `
  --save .\checkpoints\floating_grasp_decay_assist

使用增强耦合 XML 训练：
python .\train_recurrent_ppo_floating_grasp.py `
  --xml .\phy3.02_schemeA_grasp_ellipsoid.xml `
  --algo recurrentppo `
  --assist-mode off `
  --timesteps 1500000 `
  --n-envs 4 `
  --save .\checkpoints\floating_grasp_pureA_ellipsoid

- 默认 XML 耦合偏弱
- 想强化尾部-流体-base 的传递效应

恢复训练：

如果已有某个中断模型和 vecnorm 文件，可继续训练。

PowerShell：
python .\train_recurrent_ppo_floating_grasp.py `
  --xml .\phy3.02_schemeA_grasp.xml `
  --algo recurrentppo `
  --resume .\checkpoints\floating_grasp_pureA.zip `
  --vecnorm .\checkpoints\floating_grasp_pureA_vecnormalize.pkl `
  --timesteps 500000 `
  --n-envs 4 `
  --save .\checkpoints\floating_grasp_pureA_resume

回放训练结果：
python .\play_recurrent_ppo_floating_grasp.py `
  --model .\checkpoints\floating_grasp_pureA.zip `
  --deterministic



