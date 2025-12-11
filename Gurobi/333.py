

# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

# -------------------------- 参数定义（与原OPL完全一致）--------------------------
n = 3  # 工件数量（原OPL n=3）
I = range(1, n + 1)  # 工件索引 1..3
n_i = 3  # 每个工件的工序数量（原OPL n_i=3）
J_i = range(1, n_i + 1)  # 工序索引 1..3
r_f = 3  # 工厂（车间）数量（原OPL r_f=3）
F = range(1, r_f + 1)  # 工厂索引 1..3
m_ijf = 3  # 每个工厂中能处理O_ij的机器数量（原OPL为3台）
K_ijf = range(1, m_ijf + 1)  # 机器索引 1..3

# 基础参数（完全沿用原OPL定义）
AT = {1: 2, 2: 3, 3: 5}  # 工件i的到达时间
DD = {1: 10, 2: 12, 3: 14}  # 工件i的交货期
TM = 3  # 同一工厂内转运时间
TF = 6  # 不同工厂间转运时间
U = {1: 1.4, 2: 1.4, 3: 1.0}  # 工件权重（与原OPL一致）
M = 100000  # 大M参数

# -------------------------- 数据矩阵定义（严格对应你提供的CPLEX数据）--------------------------
# 维度说明：3个工厂 × 3台机器 × 3个工件 × 3道工序（车间→机器→工件→工序）

# 1. 可选机器矩阵 x_fkij[工厂f][机器k][工件i][工序j]（0/1，完全复用你的CPLEX数据）
x_cplex = [
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 工厂1-机器1（M1）
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],  # 工厂1-机器2（M2）
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]]  # 工厂1-机器3（M3）
    ],
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 工厂2-机器1（M4）
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],  # 工厂2-机器2（M5）
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]]  # 工厂2-机器3（M6）
    ],
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 工厂3-机器1（M7）
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],  # 工厂3-机器2（M8）
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]]  # 工厂3-机器3（M9）
    ]
]

# 2. 加工时间矩阵 pt_fkij[工厂f][机器k][工件i][工序j]（完全复用你的CPLEX数据）
pt_cplex = [
    [
        [[67, 56, 61], [50, 38, 7], [11, 75, 4]],  # 工厂1-机器1（M1）
        [[59, 40, 10], [33, 15, 20], [43, 7, 18]],  # 工厂1-机器2（M2）
        [[71, 57, 61], [8, 48, 34], [52, 45, 33]]  # 工厂1-机器3（M3）
    ],
    [
        [[11, 18, 4], [30, 73, 49], [33, 30, 64]],  # 工厂2-机器1（M4）
        [[13, 11, 48], [71, 64, 72], [61, 35, 44]],  # 工厂2-机器2（M5）
        [[56, 55, 50], [73, 61, 15], [61, 7, 41]]  # 工厂2-机器3（M6）
    ],
    [
        [[34, 18, 70], [7, 31, 23], [49, 6, 41]],  # 工厂3-机器1（M7）
        [[47, 47, 20], [36, 18, 35], [1, 8, 38]],  # 工厂3-机器2（M8）
        [[14, 15, 75], [27, 8, 68], [37, 46, 14]]  # 工厂3-机器3（M9）
    ]
]

# 3. 故障发生率矩阵 bd_fkij[工厂f][机器k][工件i][工序j]（完全复用你的CPLEX数据）
bd_cplex = [
    [
        [[0, 1, 0], [1, 0, 1], [0, 1, 1]],  # 工厂1-机器1（M1）
        [[1, 1, 0], [1, 1, 1], [0, 0, 1]],  # 工厂1-机器2（M2）
        [[0, 1, 0], [1, 0, 1], [1, 0, 1]]  # 工厂1-机器3（M3）
    ],
    [
        [[0, 0, 1], [1, 1, 1], [0, 1, 1]],  # 工厂2-机器1（M4）
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],  # 工厂2-机器2（M5）
        [[0, 1, 1], [1, 0, 1], [0, 1, 1]]  # 工厂2-机器3（M6）
    ],
    [
        [[1, 1, 0], [0, 0, 1], [0, 0, 0]],  # 工厂3-机器1（M7）
        [[0, 1, 1], [1, 0, 0], [0, 0, 1]],  # 工厂3-机器2（M8）
        [[1, 0, 1], [0, 0, 1], [0, 0, 1]]  # 工厂3-机器3（M9）
    ]
]

# 4. 故障维修时间矩阵 rt_fkij[工厂f][机器k][工件i][工序j]（完全复用CPLEX数据）
rt_cplex = [
    [
        [[3.7, 3.7, 3.7], [3.7, 3.7, 3.7], [3.7, 3.7, 3.7]],  # 工厂1-机器1（M1）
        [[3.6, 3.6, 3.6], [3.6, 3.6, 3.6], [3.6, 3.6, 3.6]],  # 工厂1-机器2（M2）
        [[3.4, 3.4, 3.4], [3.4, 3.4, 3.4], [3.4, 3.4, 3.4]]  # 工厂1-机器3（M3）
    ],
    [
        [[3.8, 3.8, 3.8], [3.8, 3.8, 3.8], [3.8, 3.8, 3.8]],  # 工厂2-机器1（M4）
        [[2.6, 2.6, 2.6], [2.6, 2.6, 2.6], [2.6, 2.6, 2.6]],  # 工厂2-机器2（M5）
        [[2.3, 2.3, 2.3], [2.3, 2.3, 2.3], [2.3, 2.3, 2.3]]  # 工厂2-机器3（M6）
    ],
    [
        [[2.8, 2.8, 2.8], [2.8, 2.8, 2.8], [2.8, 2.8, 2.8]],  # 工厂3-机器1（M7）
        [[2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [2.5, 2.5, 2.5]],  # 工厂3-机器2（M8）
        [[2.7, 2.7, 2.7], [2.7, 2.7, 2.7], [2.7, 2.7, 2.7]]  # 工厂3-机器3（M9）
    ]
]

# -------------------------- 数据转换：CPLEX 0-based → Gurobi 1-based 字典 --------------------------
x_fkij = {}
pt_fkij = {}
bd_fkij = {}
rt_fkij = {}

for f in F:  # 工厂1-3
    for k in K_ijf:  # 机器1-3
        for i in I:  # 工件1-3
            for j in J_i:  # 工序1-3
                # 映射CPLEX的0-based索引到Gurobi的1-based索引（严格对应数据层次）
                x_fkij[f, k, i, j] = x_cplex[f - 1][k - 1][i - 1][j - 1]
                pt_fkij[f, k, i, j] = pt_cplex[f - 1][k - 1][i - 1][j - 1]
                bd_fkij[f, k, i, j] = bd_cplex[f - 1][k - 1][i - 1][j - 1]
                rt_fkij[f, k, i, j] = rt_cplex[f - 1][k - 1][i - 1][j - 1]

# -------------------------- 创建Gurobi模型 --------------------------
model = gp.Model("JobShop_Scheduling_with_Fault_3Jobs_3Factories")

# -------------------------- 决策变量定义（与原OPL一一对应）--------------------------
X = model.addVars(I, J_i, F, K_ijf, vtype=GRB.BINARY, name="X")  # 机器分配（工件i-工序j→工厂f-机器k）
Y = model.addVars(I, J_i, I, J_i, vtype=GRB.BINARY, name="Y")  # 工序顺序（(i1,j1)在(i2,j2)之前）
Z1 = model.addVars(I, J_i, vtype=GRB.BINARY, name="Z1")  # 厂内转运（同一工厂不同机器）
Z2 = model.addVars(I, J_i, vtype=GRB.BINARY, name="Z2")  # 厂间转运（不同工厂）
B = model.addVars(I, J_i, vtype=GRB.CONTINUOUS, name="B", lb=0)  # 工序开始时间
C = model.addVars(I, vtype=GRB.CONTINUOUS, name="C")  # 工件完工时间
MT = model.addVars(I, vtype=GRB.CONTINUOUS, name="MT", lb=0)  # 工件延期时间
MTJ = model.addVar(vtype=GRB.CONTINUOUS, name="MTJ")  # 平均延期时间（优化目标）

# -------------------------- 目标函数（最小化平均延期时间，与原OPL一致）--------------------------
model.setObjective(MTJ, GRB.MINIMIZE)

# -------------------------- 约束条件（严格对应原OPL逻辑，顺序一致）--------------------------
# 约束1：MTJ上界
model.addConstr(MTJ <= M, name="MTJ_UpperBound")

# 约束2：同一工序只能分配给一台机器
for i in I:
    for j in J_i:
        model.addConstr(
            gp.quicksum(X[i, j, f, k] for f in F for k in K_ijf) <= 1,
            name=f"OneMachinePerOp_{i}_{j}"
        )

# 约束3：工序必须分配给能处理它的机器（x_fkij=1的机器）
for i in I:
    for j in J_i:
        model.addConstr(
            gp.quicksum(X[i, j, f, k] * x_fkij[f, k, i, j] for f in F for k in K_ijf) == 1,
            name=f"ValidMachineAssignment_{i}_{j}"
        )

# 约束4：转移方式互斥（仅工序2及以后）
for i in I:
    for j in J_i:
        if j >= 2:
            # Z1 + Z2 <= 1 - 同一机器（无转运）
            same_machine = gp.quicksum(X[i, j, f, k] * X[i, j - 1, f, k] for f in F for k in K_ijf)
            model.addConstr(Z1[i, j] + Z2[i, j] <= 1 - same_machine, name=f"TransferMutualExcl_{i}_{j}_1")

            # Z1 >= 同一工厂不同机器（厂内转运）
            same_factory_diff_machine = gp.quicksum(
                X[i, j, f, k] * X[i, j - 1, f, k1]
                for f in F for k in K_ijf for k1 in K_ijf if k != k1
            )
            model.addConstr(Z1[i, j] >= same_factory_diff_machine, name=f"TransferZ1_{i}_{j}")

            # Z2 >= 不同工厂（厂间转运）
            same_factory = gp.quicksum(
                X[i, j, f, k] * X[i, j - 1, f, k1]
                for f in F for k in K_ijf for k1 in K_ijf
            )
            model.addConstr(Z2[i, j] >= 1 - same_factory, name=f"TransferZ2_{i}_{j}")

# 约束5：第一道工序无转移
for i in I:
    model.addConstr(Z1[i, 1] + Z2[i, 1] == 0, name=f"NoTransferFirstOp_{i}")

# 约束6：第一道工序开始时间 >= 到达时间
for i in I:
    model.addConstr(B[i, 1] >= AT[i], name=f"FirstOpStartTime_{i}")

# 约束7：同一工件的先后工序约束
for i in I:
    for j in J_i:
        if j < n_i:
            prev_process_time = gp.quicksum(
                X[i, j, f, k] * (pt_fkij[f, k, i, j] + rt_fkij[f, k, i, j] * bd_fkij[f, k, i, j])
                for f in F for k in K_ijf
            )
            model.addConstr(
                B[i, j + 1] >= B[i, j] + prev_process_time + Z1[i, j + 1] * TM + Z2[i, j + 1] * TF,
                name=f"SameJobPrecedence_{i}_{j}"
            )

# 约束8：同一机器上的工序不重叠
for i1 in I:
    for i2 in I:
        for j1 in J_i:
            for j2 in J_i:
                if j1 != j2:
                    for f in F:
                        for k in K_ijf:
                            # 工序(i1,j1)完工时间 ≤ 工序(i2,j2)开始时间
                            process_time1 = X[i1, j1, f, k] * (
                                        pt_fkij[f, k, i1, j1] + rt_fkij[f, k, i1, j1] * bd_fkij[f, k, i1, j1])
                            lhs1 = B[i1, j1] + process_time1
                            rhs1 = B[i2, j2] + M * (3 - Y[i1, j1, i2, j2] - X[i1, j1, f, k] - X[i2, j2, f, k])
                            model.addConstr(lhs1 <= rhs1, name=f"MachineNoOverlap1_{i1}_{j1}_{i2}_{j2}_{f}_{k}")

                            # 工序(i2,j2)完工时间 ≤ 工序(i1,j1)开始时间
                            process_time2 = X[i2, j2, f, k] * (
                                        pt_fkij[f, k, i2, j2] + rt_fkij[f, k, i2, j2] * bd_fkij[f, k, i2, j2])
                            lhs2 = B[i2, j2] + process_time2
                            rhs2 = B[i1, j1] + M * (2 + Y[i1, j1, i2, j2] - X[i1, j1, f, k] - X[i2, j2, f, k])
                            model.addConstr(lhs2 <= rhs2, name=f"MachineNoOverlap2_{i1}_{j1}_{i2}_{j2}_{f}_{k}")

# 约束9：工件完工时间计算
for i in I:
    last_process_time = gp.quicksum(
        X[i, n_i, f, k] * (pt_fkij[f, k, i, n_i] + rt_fkij[f, k, i, n_i] * bd_fkij[f, k, i, n_i])
        for f in F for k in K_ijf
    )
    model.addConstr(C[i] >= B[i, n_i] + last_process_time, name=f"JobCompletionTime_{i}")

# 约束10：工序开始时间非负
for i in I:
    for j in J_i:
        model.addConstr(B[i, j] >= 0, name=f"StartTimeNonNegative_{i}_{j}")

# 约束11：延期时间计算（MT[i] = max(C[i]-DD[i], 0)）
for i in I:
    model.addConstr(MT[i] >= C[i] - DD[i], name=f"MakeSpanLower1_{i}")
    model.addConstr(MT[i] >= 0, name=f"MakeSpanLower2_{i}")

# 约束12：平均延期时间定义（加权平均）
model.addConstr(
    MTJ >= gp.quicksum(MT[i] * U[i] for i in I) / n,
    name="AverageMakeSpan"
)

# -------------------------- 求解参数设置（适配gurobipy 9.5.0）--------------------------
model.Params.LogToConsole = 1  # 显示求解日志
model.Params.TimeLimit = 7200  # 求解时间限制（2小时）
model.Params.MIPGap = 0.01  # 允许1%相对误差（加速求解）
model.Params.FeasibilityTol = 1e-6  # 可行性容差（避免数值误差）

# -------------------------- 求解与结果输出 --------------------------
model.optimize()

if model.status == GRB.OPTIMAL:
    print("\n==================== 最优解结果 ====================")
    print(f"最小平均延期时间: {MTJ.x:.2f}")
    print("\n各工件详细信息：")
    for i in I:
        print(f"\n工件{i}：")
        print(f"  到达时间: {AT[i]}, 交货期: {DD[i]}")
        print(f"  完工时间: {C[i].x:.2f}, 延期时间: {MT[i].x:.2f}")
        print("  工序分配详情：")
        for j in J_i:
            for f in F:
                for k in K_ijf:
                    if X[i, j, f, k].x > 0.5:  # 找到分配的机器
                        start_time = B[i, j].x
                        process_time = pt_fkij[f, k, i, j]
                        fault_time = rt_fkij[f, k, i, j] * bd_fkij[f, k, i, j]
                        total_time = process_time + fault_time
                        transfer_type = "无转运" if j == 1 else ("厂内转运" if Z1[i, j].x > 0.5 else "厂间转运")
                        original_machine = 3 * (f - 1) + k  # 还原原CPLEX机器号（M1-M9）
                        print(
                            f"    工序{j}: 工厂{f}机器{k}（原M{original_machine}） | 开始时间: {start_time:.2f} | 加工时间: {process_time} | 故障时间: {fault_time:.1f} | {transfer_type}")
elif model.status == GRB.TIME_LIMIT:
    print(f"\n==================== 求解超时结果 ====================")
    print(f"当前最佳平均延期时间: {MTJ.x:.2f}")
    print(f"相对误差: {model.MIPGap:.2%}")
elif model.status == GRB.INFEASIBLE:
    print("\n模型无可行解！")
    model.computeIIS()
    model.write("infeasible_model.ilp")
    print("不可行约束已保存到 infeasible_model.ilp 文件，可用于排查原因")
else:
    print(f"\n模型求解失败，状态码: {model.status}")
    print("状态说明：", GRB.StatusToString(model.status))
