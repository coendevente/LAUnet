import numpy as np
import matplotlib.pyplot as plt

# results = np.array([
#     [0.3835598149453045, 0.4150208200667467, 0.08865246598812192, 0.3455573518925075],
#     [0.4441841334723698, 0.41860619504716995, 0.4425503869407868, 0.4331938667008088],
#     [0.41161859449705945, 0.4821400757093977, 0.46172426778727543, 0.5019622215159906]
# ])
results = np.array([
    [0.3540552137956657, 0.3830961416000739, 0.08183467737129707, 0.31897601713154544],
    [0.4100161232052644, 0.38640571850507993, 0.4085080494838032, 0.3998712615699773],
    [0.37995562568959335, 0.4450523775779056, 0.4262070164190235, 0.4633497429378375]
])

std = np.std(results, axis=1)
mean = np.mean(results, axis=1)
print('std == {}'.format(std))
print('mean == {}'.format(std))

x = [1, 2, 3]
# plt.scatter(x, results)
thickness = 2
markersize = 7
fs = 15
plt.figure(figsize=(4, 6))
plt.errorbar(x, mean, yerr=std, fmt='o', capsize=10, elinewidth=thickness, capthick=thickness, markersize=markersize)
plt.ylabel('Dice', fontsize=fs)
plt.xlabel('Architecture', fontsize=fs)
plt.xlim([0, 4])
plt.ylim([min([mean[i] - std[i] - .2 for i in range(len(x))]), max([mean[i] + std[i] + .02 for i in range(len(x))])])
plt.xticks([1, 2, 3], fontsize=fs)
plt.yticks(fontsize=fs)
for i in range(len(x)):
    m, s = round(mean[i], 3), round(std[i], 3)
    plt.text(x[i] - .15, mean[i] - std[i] - .03, '${} \pm {}$'.format(m, s), rotation=-90, fontsize=fs)
plt.show()