import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
# ours = np.loadtxt("log/fl_200.txt")
fl_noise_1 = np.loadtxt("dp_mnist0.5.txt")
fl_noise_2 = np.loadtxt("dp_mnist1.0.txt")
fl_noise_3 = np.loadtxt("dp_mnist1.5.txt")
ours = np.loadtxt("val_mnist2.0.txt")
# plot these accuracy in the same plot
fig, ax = plt.subplots()
ax.plot(ours, label="Ours")
ax.plot(fl_noise_1, label="noise level = 0.5")
ax.plot(fl_noise_2, label="noise level = 1")
ax.plot(fl_noise_3, label="noise level = 1.5")
# ax.plot(cl, label="CL")
ax.set(xlabel='Round', ylabel='Accuracy',
       )
plt.title("MNIST")
# ax.grid(True)
plt.legend()
plt.show()
# fig.savefig("noise2.png")

# byzantine = np.loadtxt("log/fl_200.txt")
# no_byzantine = np.loadtxt('log/vanilla_acc_200.txt')
# # plot these accuracy in the same plot
# fig, ax = plt.subplots()
# ax.plot(no_byzantine, label="No Byzantine")
# ax.plot(byzantine, label="Byzantine")
# ax.set(xlabel='Round', ylabel='Accuracy',
#        )
# # ax.grid(True)
# plt.legend()
# plt.title("CIFAR10")
# # plt.show()
# fig.savefig("byzantine3.png")


# save the plot
# fig.savefig("acc4.png")
# fig.savefig("byzantine2.png")
