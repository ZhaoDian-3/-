import neal
from pyqubo import Binary, Constraint

# 定义变量
x1, x2 = Binary('x1'), Binary('x2')

# 约束强度
M = 5.0

# 定义哈密顿算符H
H = 3 * x1**2 - 2 * x2**2 + M * Constraint((x2 - x1*x2), label='x1>=x2')

# 编译模型
model = H.compile()

# 转换为QUBO格式
qubo, offset = model.to_qubo()

# 使用模拟退火求解器
sampler = neal.SimulatedAnnealingSampler()
raw_solution = sampler.sample_qubo(qubo)

# 提取并打印解
for sample in raw_solution:
    print(sample)
