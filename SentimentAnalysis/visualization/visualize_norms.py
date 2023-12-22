import numpy as np
import seaborn as sns
import numpy
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.linear_model import BayesianRidge
from scipy.stats import pearsonr
font = {'family': 'serif',
            'size': 15,
            'serif': 'cmr10'
            }
mpl.rc('font', **font)
mpl.rc('legend', fontsize=15)
mpl.rc('axes', labelsize=19)
map_color = 'tab:red'
stochastic_color = 'tab:green'
point_err_color = 'tab:blue'
plt.rcParams['axes.unicode_minus'] = False



class VisualizeModuleSelection:

    def __init__(self, path_to_norms_and_names):

        self.norms_and_names = pickle.load(open(path_to_norms_and_names, 'rb'))
        self.num_modules = [2, 3, 4, 5, 8, 11, 13,17]

    def plot_boxplots(self):

        norms = np.array([val['norms'] for val in self.norms_and_names.values()])
        norms = np.mean(norms, 0)
        names = [val['names'] for val in self.norms_and_names.values()][0]

        df = pd.DataFrame()
        df['Operator norm'] = norms
        df['Module type'] = ['Attention' if 'attention' in name else 'MLP' for name in names]
        fig, ax = plt.subplots(1,1)
        sns.violinplot(data =df, x = 'Module type', y = 'Operator norm',hue = 'Module type' ,ax = ax)
        plt.gcf().subplots_adjust(bottom=0.12)
        ax.set_title('Operator norms for Attention & MLP')
        fig.savefig(r'C:\Users\45292\Documents\Master\SentimentClassification\IMDB Figures\Model General\operator_norm_violin.pdf', format = 'pdf')
        plt.show()

    def get_and_plot(self, minimum = False, ax = None):

        runs = []
        num_pairs = []
        num_modules = []
        for num in self.num_modules:
            for run in range(5):
                names, norms = self.get_selection(run, num, restriction='attention', minimum = minimum)
                modules = self.get_modules(names, norms)
                cnt_pairs = self.count_pairs(modules)

                num_pairs.append(cnt_pairs)
                runs.append(run)
                num_modules.append(num)

        df = pd.DataFrame()
        df['Modules'] = num_modules
        df['Pairs'] = num_pairs
        if ax is None:
            fig, ax = plt.subplots(1,1)
        label = 'Attn module pairs min norm' if minimum else 'Attn module pairs max norm'
        sns.pointplot(errorbar=lambda x: np.percentile(x, [25, 75]),
                      data=df, x="Modules", y='Pairs',
                      join=False,
                      capsize=.30,
                      markers="d",
                      scale=1.0,
                      err_kws={'linewidth': 0.7}, estimator=np.median,
                      color='tab:orange' if minimum else 'tab:blue',
                      label=label,
                      ax=ax)

        return ax

    def make_restriction(self, names, norms, restriction):
        names_, norms_ = [], []

        if restriction == 'both':
            return names, norms
        for name, norm in zip(names, norms):
            if restriction in name:
                names_.append(name)
                norms_.append(norm)

        return names_, norms_


    def get_selection(self, run_number, num, restriction = 'both', minimum = False):


        norms = self.norms_and_names[run_number]['norms']
        names = self.norms_and_names[run_number]['names']
        names, norms = self.make_restriction(names, norms, restriction)

        argsorted = np.argsort(norms)
        if not minimum:
            argsorted = argsorted[::-1]
        names = [names[i] for i in argsorted[:num]]
        norms = [norms[i] for i in argsorted[:num]]
        return names, norms

    def get_modules(self, names, norms, modules=None):

        if modules is None:
            modules = {}
        for name, norm in zip(names, norms):
            if 'layer' in name:
                number = int(name.split(".")[4])
                if number not in modules:
                    modules[number] = {'names': [], 'norms': []}
                modules[number]['norms'].append(norm)
                modules[number]['names'].append(name)

        return modules

    def count_pairs(self, module):
        num_pairs = 0
        for key, val in module.items():
            names = [name for name in val['names'] if 'out_lin' not in name]
            num_pairs += max((len(names)-1, 0))

        return num_pairs


    def plot_number_of_pairs(self, modules):
        pass


class Simulation:

    def __init__(self, data_noise = 1, weight_noise = 1, sample_noise = 1):
        self.X = np.random.normal(0,data_noise, (100, 10))
        self.beta  = np.random.normal(0,weight_noise, (10, 1))
        self.y = self.X @ self.beta
        self.y = self.y.ravel() + np.random.normal(0, sample_noise, (100, ))
    def train_and_get_weights_and_variances(self, lambda_1 = 1e-6, lambda_2=1e-6, alpha_1=1e-6, alpha_2=1e-6):
        regressor = BayesianRidge(lambda_1 = lambda_1, lambda_2 = lambda_2, alpha_1=alpha_1,alpha_2=alpha_2).fit(self.X, self.y)
        coefs, sigma = regressor.coef_, regressor.sigma_
        return coefs, sigma

class Simulate:

    def __init__(self, num_simulations = 100, alpha_range = (1e-6,1e-5), lambda_range = (1e-6,1e-5)):

        self.num_simulations = num_simulations
        self.alpha_range = np.linspace(*alpha_range, num = num_simulations)
        self.lambda_range = np.linspace(*lambda_range, num = num_simulations)
        self.data_noise_range = np.linspace(0.1, 2, num_simulations)
        self.beta_noise_range = np.linspace(0.1, 2, num_simulations)
        self.sample_noise_range = np.linspace(0.1, 2, num_simulations)

    def simulate(self):
        results = []
        coefs = []
        variances = []
        counter = 0
        max_ = self.num_simulations ** 5
        for alpha in self.alpha_range:
            for lambda_ in self.lambda_range:
                for data_noise in self.data_noise_range:
                    for beta_noise in self.beta_noise_range:
                        for sample_noise in self.sample_noise_range:
                            simulation = Simulation(data_noise, beta_noise, sample_noise)
                            coef, sigma = simulation.train_and_get_weights_and_variances(lambda_,lambda_,alpha, alpha)

                            coefs+= list(coef)
                            variances +=list(sigma.diagonal())
                            if (counter +1) % 100 == 0:
                                print(f"Finished {counter} simulations out of {max_}")
                            counter +=1
        correlation = pearsonr(np.abs(coefs), variances)
        df = pd.DataFrame()
        df['l1 norms'] = np.abs(coefs)
        df['Variance'] = variances
        sns.regplot(data=df, x="l1 norms", y="Variance")
        plt.show()
        print(correlation)

    def simulate_(self):
        coefs, variances, results = [], [],[]

        for i in range(400):
            simulation = Simulation()
            coef, sigma = simulation.train_and_get_weights_and_variances()
            coefs+= list(coef)
            variances += list(sigma.diagonal())
            results.append(pearsonr(np.abs(coef), sigma.diagonal()))

        df = pd.DataFrame()
        df['l1 norms'] = np.abs(coefs)
        df['Variance'] = variances
        sns.regplot(data=df, x="l1 norms", y="Variance")
        plt.show()
        print(pearsonr(np.abs(coefs), variances))
        print(np.mean(results))
        return results

if __name__ == '__main__':

    simulator = Simulate(num_simulations=6)
    results = simulator.simulate()
    breakpoint()


    plotter = VisualizeModuleSelection(r"C:\Users\45292\Documents\Master\SentimentClassification\imdb_dataset\norms_and_names.pkl")
    fig, ax = plt.subplots(1,1)
    ax = plotter.get_and_plot(minimum=True, ax = ax)
    ax = plotter.get_and_plot(minimum=False, ax = ax )
    ylims = ax.get_ylim()
    additional_top = (ylims[1] - ylims[0]) * 0.3 + ylims[1]
    ax.set_ylim((ylims[0], additional_top))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
              ncol=1, fancybox=True, shadow=True)
    ax.set_title('Num. attention module (Q,K) pairs selected')
    ax.set_xlabel('Num. Modules')
    ax.set_ylabel('Num. Pairs')
    fig.savefig(r'C:\Users\45292\Documents\Master\SentimentClassification\IMDB Figures\Model General\num_qkv_pairs.pdf', format = 'pdf')

    plt.show()

