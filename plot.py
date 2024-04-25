import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
ours = np.loadtxt("log/vanilla_acc.txt")
fl_noise_1 = np.loadtxt("log/noise_0.5.txt")
fl_noise_2 = np.loadtxt("log/noise_1.txt")
fl_noise_3 = np.loadtxt("log/noise_1.5.txt")
cl = np.loadtxt("log/centralized_acc.txt")
# plot these accuracy in the same plot
fig, ax = plt.subplots()
ax.plot(ours, label="Ours")
ax.plot(fl_noise_1, label="noise level = 0.5")
ax.plot(fl_noise_2, label="noise level = 1")
ax.plot(fl_noise_3, label="noise level = 1.5")
ax.plot(cl, label="CL")
ax.set(xlabel='Round', ylabel='Accuracy',
       )
ax.grid(True)
plt.legend()
# plt.show()


# save the plot
fig.savefig("acc3.png")
# fig.savefig("byzantine.png")
