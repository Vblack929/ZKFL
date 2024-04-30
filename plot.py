import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
ours = np.loadtxt("log/fl_200.txt")
fl_noise_1 = np.loadtxt("log/noise05.txt")
fl_noise_2 = np.loadtxt("log/noise1.txt")
fl_noise_3 = np.loadtxt("log/noise_15.txt")
# cl = np.loadtxt("cl_200.txt")
# plot these accuracy in the same plot
fig, ax = plt.subplots()
ax.plot(ours, label="Ours")
ax.plot(fl_noise_1, label="noise level = 0.5")
ax.plot(fl_noise_2, label="noise level = 1")
ax.plot(fl_noise_3, label="noise level = 1.5")
# ax.plot(cl, label="CL")
ax.set(xlabel='Round', ylabel='Accuracy',
       )
ax.grid(True)
plt.legend()
plt.show()

# no_byzantine = np.loadtxt("zk_nomal.txt")
# byzantine = np.loadtxt('zk_mal.txt')
# # plot these accuracy in the same plot
# fig, ax = plt.subplots()
# ax.plot(no_byzantine, label="No Byzantine")
# ax.plot(byzantine, label="Byzantine")
# ax.set(xlabel='Round', ylabel='Accuracy',
#        )
# ax.grid(True)
# plt.legend()
# plt.show()


# save the plot
# fig.savefig("acc4.png")
# fig.savefig("byzantine2.png")
