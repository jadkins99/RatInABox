import matplotlib.pyplot as plt

def plot_dead_neurons_over_time(x,y,x_label,y_label):

    plt.figure(figsize=(8,4))
    plt.plot(x, y, color='red', lw=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Remove grid
    plt.grid(False)

    # Remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()