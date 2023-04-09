from matplotlib import pyplot as plt
import numpy as np
import torch



X = [100, 200, 500, 1000, 2000]
Y1 = [69.43, 72.41,65.04, 60.03,65.599]
Y2 = [2.6463, 2.637,2.60, 2.6204, 2.611]
Y3=  [None,None,55.675,57.69,54.742]

X_p2 = ['(4,128)', '(4,256)', '(5,256)', '(5,128)']
Y_p2_sin = [61.86,60.03,68.26,None]
Y_p2_helix = [59.69,60.63,51.32,62]

X_p3 = [10, 50, 100, 150, 200]
Y_p3_sin = [67.43,68.82,74.9,69.76,72.51][-1::-1]
Y_p3_helix = [61.41,48.51,49.97,48.57,50.12]

X_p4 = [ '0.005','0.01','0.02','cosine','sigmoid','cos_square']
Y_p4_sin = [62.92,60.47,61.34,63.30,63.85,63.88]
Y_p4_helix = [49.403,55.48,49.2,54.26,67.12,53.83]



# plt.plot(X, Y1, '-o', label='Sine',)
# plt.plot(X, Y2, label='label2')
# plt.plot(X, Y3, '-o',qlabel='Helix')
# plt.plot(X, Y4, label='label4')

# plt.plot(X_p2, Y_p2_sin, '-o', label='Sine')
# plt.plot(X_p2, Y_p2_helix, '-o', label='Helix')

# plt.plot(X_p3, Y_p3_sin, '-o', label='Sine')
# plt.plot(X_p3, Y_p3_helix, '-o', label='Helix')

plt.plot(X_p4, Y_p4_sin, '-o', label='Sine')
plt.plot(X_p4, Y_p4_helix, '-o', label='Helix')


plt.ylabel('EMD Score')
plt.xlabel('Noise Schedule (ubeta) for linear')
plt.title(label='EMD Score vs Noise Schedule')

plt.legend()
plt.savefig('plot.png')
plt.show()