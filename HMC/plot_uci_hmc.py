import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_nll_results(nll_scaled_boston = [], nll_scaled_energy = [], nll_scaled_yacht = []):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data and set labels for each dataset
    ax.plot(nll_scaled_boston, label='Boston')
    ax.plot(nll_scaled_energy, label='Energy')
    ax.plot(nll_scaled_yacht, label='Yacht')

    # Add legend
    ax.legend()

    # Set axis labels and title
    # ax.set_xlabel('Iteration')
    ax.set_ylabel('Negative Log Likelihood')
    ax.set_title('Negative Log Likelihood for Scaled Datasets')

    # Show the plot
    plt.show()

if __name__ == '__main__':
    boston_results = pickle.load(open(r'C:\Users\Gustav\Desktop\MasterThesis\UCI_HMC\boston_models\boston_scaled.pkl', 'rb'))
    energy_results = pickle.load(open(r'C:\Users\Gustav\Desktop\MasterThesis\UCI_HMC\energy_models\energy_scaled.pkl', 'rb'))
    yacht_results = pickle.load(open(r'C:\Users\Gustav\Desktop\MasterThesis\UCI_HMC\yacht_models\yacht_scaled.pkl', 'rb'))

    boston_results_not_scaled = pickle.load(open(r'C:\Users\Gustav\Desktop\MasterThesis\UCI_HMC\boston_models\boston_not_scaled.pkl', 'rb'))
    energy_results_not_scaled = pickle.load(open(r'C:\Users\Gustav\Desktop\MasterThesis\UCI_HMC\energy_models\energy_not_scaled.pkl', 'rb'))
    yacht_results_not_scaled = pickle.load(open(r'C:\Users\Gustav\Desktop\MasterThesis\UCI_HMC\yacht_models\yacht_not_scaled.pkl', 'rb'))

    nll_scaled_boston = [-b['test_ll'] for b in boston_results['all_results_scaled']]
    nll_scaled_energy = [-e['test_ll'] for e in energy_results['all_results_scaled']]
    nll_scaled_yacht = [-y['test_ll'] for y in yacht_results['all_results_scaled']]

    nll_not_scaled_boston = [-b['test_ll'] for b in boston_results_not_scaled['all_results_not_scaled']]
    nll_not_scaled_energy = [-e['test_ll'] for e in energy_results_not_scaled['all_results_not_scaled']]
    nll_not_scaled_yacht = [-y['test_ll'] for y in yacht_results_not_scaled['all_results_not_scaled']]

    plot_nll_results(nll_scaled_boston, nll_scaled_energy, nll_scaled_yacht)
    plot_nll_results(nll_not_scaled_boston, nll_not_scaled_energy, nll_not_scaled_yacht)
    breakpoint()

