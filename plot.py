import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
fl_dp = np.loadtxt('fl_dp.txt')
fl_without_dp = np.loadtxt('vanilla_acc.txt')
cl = np.loadtxt('centralized_acc.txt')

# plot these accuracy in the same plot
fig, ax = plt.subplots()
ax.plot(fl_dp, label='FL with DP')
ax.plot(fl_without_dp, label='FL without DP')
ax.plot(cl, label='Centralized')
ax.set(xlabel='Epochs', ylabel='Accuracy',
       title='Accuracy of FL with and without DP')
ax.legend()
ax.grid()
plt.show()

# save the plot
fig.savefig("acc.png")
