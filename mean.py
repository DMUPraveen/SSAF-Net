import numpy as np
from AE_PPM_main import train

all_step = 5  # 总迭代次数

# cases = ['ex2', 'ridge', 'urban', 'houston', 'synthetic']
# case = cases[1]

# 存放独立运行结果
armse_y_all = []
asad_y_all = []
armse_a_all = []
armse_em_all = []
asad_em_all = []
tim_all = []
'''
aRMSE_Y: 1.90 ± 0.12
aSAD_Y: 3.34 ± 0.17
aRMSE_a: 2.25 ± 0.47
aRMSE_M: 0.26 ± 0.14
aSAD_em: 0.44 ± 0.24'''
for i in range(all_step):
    print('traning on', i + 1)
    armse_y, asad_y, armse_a, armse_em, asad_em, tim= train()

    armse_y_all.append(armse_y)
    armse_a_all.append(armse_a)
    asad_y_all.append(asad_y)
    tim_all.append(tim)
    armse_em_all.append(armse_em)
    asad_em_all.append(asad_em)

    print('armse_a, armse_em', armse_a, armse_em)
    print('-' * 50)

print('*' * 70)
print('RESULTS:')
print('step:', all_step)
print(f'time elapsed: {np.mean(tim):.2f}')
print(f'aRMSE_Y: {np.mean(armse_y_all) * 100:.2f} ± {np.std(armse_y_all) * 100:.2f}')
print(f'aSAD_Y: {np.mean(asad_y_all) * 100:.2f} ± {np.std(asad_y_all) * 100:.2f}')
print(f'aRMSE_a: {np.mean(armse_a_all) * 100:.2f} ± {np.std(armse_a_all) * 100:.2f}')
print(f'aRMSE_M: {np.mean(armse_em_all) * 100:.2f} ± {np.std(armse_em_all) * 100:.2f}')
print(f'aSAD_em: {np.mean(asad_em_all) * 100:.2f} ± {np.std(asad_em_all) * 100:.2f}')
