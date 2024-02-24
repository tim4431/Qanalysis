from Qanalysis.fitting.helper_tools import *
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class FrequencyDomain:
    def __init__(self, freq, signal):
        # initialize parameters
        self.frequency = freq
        self.signal = signal
        self.n_pts = len(self.signal)
        self.is_analyzed = False
        self.p0 = None
        self.popt = None
        self.pcov = None

    def _guess_init_params(self):
        """
        Guess initial parameters from data. Will be overwritten in subclass
        """

    def _set_init_params(self, p0):
        if p0 is None:
            self._guess_init_params()
        else:
            self.p0 = p0

    def _save_fit_results(self, popt, pcov):
        self.popt = popt
        self.pcov = pcov
        self.r2_score = r2_score(
            self.signal, self.fit_func(self.frequency, *(self.popt))
        )

    def analyze(self, p0=None, plot=True, **kwargs):
        """
        Analyze the data with initial parameter `p0`.
        """
        # set initial fit parameters
        self._set_init_params(p0)
        # perform fitting
        popt, pcov = curve_fit(
            self.fit_func, self.frequency, self.signal, p0=self.p0, **kwargs
        )
        self.is_analyzed = True

        # save fit results
        self._save_fit_results(popt, pcov)

        if plot:
            self.plot_result()

    def _plot_base(self):
        fig = plt.figure()

        # plot data
        _, self.frequency_prefix = number_with_si_prefix(np.max(np.abs(self.frequency)))
        self.frequency_scaler = si_prefix_to_scaler(self.frequency_prefix)

        plt.plot(
            self.frequency / self.frequency_scaler,
            self.signal,
            ".",
            label="Data",
            color="black",
        )
        plt.xlabel("Frequency (" + self.frequency_prefix + "Hz)")
        plt.ylabel("Signal")
        plt.legend(loc=0, fontsize=14)

        fig.tight_layout()
        return fig

    def plot_result(self):
        """
        Will be overwritten in subclass
        """
        if not self.is_analyzed:
            raise ValueError("The data must be analyzed before plotting")

    def _get_const_baseline(self, baseline_portion=0.2, baseline_ref="symmetric"):

        samples = self.signal

        N = len(samples)
        if baseline_ref == "left":
            bs = np.mean(samples[: int(baseline_portion * N)])
        elif baseline_ref == "right":
            bs = np.mean(samples[int(-baseline_portion * N) :])
        elif baseline_ref == "symmetric":
            bs_left = np.mean(samples[int(-baseline_portion * N / 2) :])
            bs_right = np.mean(samples[: int(baseline_portion * N / 2)])
            bs = np.mean([bs_left, bs_right])

        return bs
