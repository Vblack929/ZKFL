import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
zkfl_by = np.loadtxt('zkfl_acc_with_mal.txt')
zkfl_no_by = np.loadtxt('zkfl_acc_without_mal.txt')

# plot these accuracy in the same plot
fig, ax = plt.subplots()
ax.plot(zkfl_by, label='with Byzantine attack')
ax.plot(zkfl_no_by, label='without Byzantine attack')
ax.set(xlabel='Epochs', ylabel='Accuracy',
       title='Accuracy of FL with and without Byzantine attack')
ax.legend()
ax.grid()
plt.show()

# save the plot
# fig.savefig("acc.png")
fig.savefig("byzantine.png")
