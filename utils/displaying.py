import matplotlib.pyplot as plt


def display_values(train_values, validation_values, type_of_plot='Loss'):
    epochs = range(1, len(train_values) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_values, 'ro-', label=f'Training {type_of_plot}')
    plt.plot(epochs, validation_values, 'bo-', label=f'Validation {type_of_plot}')
    plt.title(f'Training and Validation {type_of_plot}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{type_of_plot}')
    plt.grid(True)
    plt.legend()

    plt.savefig(f'./images/{type_of_plot}_plot.png')

    plt.show()

