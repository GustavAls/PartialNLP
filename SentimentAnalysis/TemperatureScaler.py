
import numpy as np
import torch.nn as nn
import torch
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import calibration as cal
class TemperatureScaler(nn.Module):

    def __init__(self, predictions, labels):
        super(TemperatureScaler, self).__init__()

        self.predictions = predictions
        self.labels = labels

        self.temperature = nn.Parameter(torch.ones(1)*1.2)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD([self.temperature], lr = 0.01)

    def forward(self, x):
        temp = self.temperature.unsqueeze(1).expand(1, x.shape[-1])
        out = x / temp
        return out

    def optimize(self, num_iterations):
        pbar = tqdm(range(num_iterations), desc = 'Optimizing temperature scaling')
        for niter in pbar:
            self.optimizer.zero_grad()
            out = self.forward(self.predictions)
            loss = self.loss_function(out, self.labels)
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix({'NLL': loss.item()})



class ECE:

    def __init__(self, predictions, labels, n_bins = 20, strategy = 'quantile'):

        if isinstance(predictions, torch.Tensor):
            self.predictions = predictions.detach().numpy()
        elif isinstance(predictions, np.ndarray):
            self.predictions = predictions
        else:
            raise ValueError("")

        if isinstance(labels, torch.Tensor):
            self.labels = labels.detach().numpy()
        elif isinstance(labels, np.ndarray):
            self.labels = labels
        else:
            raise ValueError("")

        self.calibration_error = cal.get_calibration_error(self.predictions, self.labels, p = 1)
        self.y_prob_zero, self.y_prob_one = self.predictions[:, 0], self.predictions[:, 1]
        if strategy == 'quantile':
            quantiles = np.linspace(0, 1, n_bins + 1)
            self.bins_zero = np.percentile(self.y_prob_zero, quantiles * 100)
            self.bins_one = np.percentile(self.y_prob_one, quantiles * 100)
        elif strategy == 'uniform':
            self.bins_one = np.linspace(0.0, 1.0, n_bins + 1)
            self.bins_zero = self.bins_one.copy()

    def calculate_for_both(self):
        class_one = self.calculate_for_class(0)
        class_two = self.calculate_for_class(1)
        return (class_one + class_two) / 2

    def calculate_for_class(self, class_num = 0):

        if class_num == 0:
            labels = (self.labels + 1) % 2
            probs = self.y_prob_zero
            bins = self.bins_zero

        elif class_num == 1:
            labels = self.labels
            probs = self.y_prob_one
            bins = self.bins_one

        else:
            raise ValueError("")

        binids = np.searchsorted(bins[1:-1], probs)
        summer = 0
        for bin in np.unique(binids):
            indices = binids == bin
            lab = labels[indices]
            prob = probs[indices]
            all_probs = self.predictions[indices]
            num_elements = np.sum(indices)/len(binids)
            acc = accuracy_score(lab, all_probs.argmax(-1))
            conf = np.mean(prob)
            summer += num_elements * np.abs(acc - conf)
        return summer


if __name__ == '__main__':

    pcl = pickle.load(open(r"C:\Users\45292\Documents\Master\NLP\SST2\laplace\sublayer_full\run_0\run_number_0.pkl", 'rb'))
    pcl = pcl['results']['1.0']
    eces = []
    for n_bins in range(5, 40, 4):
        ece = ECE(pcl.predictions, pcl.labels, n_bins, strategy='uniform')
        ec = ece.calculate_for_class(1)
        eces.append(ece.calibration_error)

    breakpoint()
