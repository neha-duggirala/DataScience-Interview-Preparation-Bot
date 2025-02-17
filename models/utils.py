import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def create_line_plot(x,y):
    # Create a line plot
    sns.lineplot(x=x, y=y, color = "green")

    # Show the plot
    plt.show()

def plot_losses(train_losses):
    # Create a dataframe to hold the losses
    df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Training Loss': train_losses,
        
    })

    # Melt the dataframe for easier plotting with seaborn
    df_melted = df.melt('Epoch', var_name='Loss Type', value_name='Loss')

    # Create the line plot
    sns.lineplot(data=df_melted, x='Epoch', y='Loss', hue='Loss Type')

    # Show the plot
    plt.show()
