from base_freq_domain import *


def gaussian_fit_func(f: float, f0: float, a: float, c: float, d: float):
    return a * np.exp(-((f - f0) ** 2) / (2 * c**2)) + d


def analyze_gaussian(f, sig, p0=None):
    sig_mag = sig
    if sig.dtype == complex:
        sig_mag = np.abs(sig) ** 2

    if p0 is None:
        if (np.max(sig_mag) - np.mean(sig_mag)) > (np.mean(sig_mag) - np.min(sig_mag)):
            # peak detected case
            f0 = f[np.argmax(sig_mag)]
            d = np.mean(
                sig_mag[np.argsort(sig_mag)[: int(len(sig_mag) // 10)]]
            )  # baseline (average of smallest 10% samples)
            # linewidth is extracted from sample closest to half-max
            c = (
                1
                / np.sqrt(2)
                * np.abs(
                    f[
                        np.argmin(
                            np.abs(sig_mag - ((np.max(sig_mag) - d) / np.exp(1) + d))
                        )
                    ]
                    - f0
                )
            )
            a = np.max(sig_mag) - d

            p0 = [f0, a, c, d]
        elif (np.max(sig_mag) - np.mean(sig_mag)) < (
            np.mean(sig_mag) - np.min(sig_mag)
        ):
            # valley detected case
            f0 = f[np.argmin(sig_mag)]
            d = np.mean(
                sig_mag[np.argsort(-sig_mag)[: int(len(sig) // 10)]]
            )  # baseline (average of largest 10% samples)
            # linewidth is extracted from sample closest to half-max
            c = (
                1
                / np.sqrt(2)
                * np.abs(
                    f[
                        np.argmin(
                            np.abs(sig_mag - ((np.min(sig_mag) - d) / np.exp(1) + d))
                        )
                    ]
                    - f0
                )
            )
            a = np.min(sig_mag) - d

            p0 = [f0, a, c, d]

        fit = curve_fit(
            gaussian_fit_func,
            f,
            sig_mag,
            p0=p0,
            bounds=(
                [p0[0] * 0.5, p0[1] * 0.5, p0[2] * 0.1, 0],
                [p0[0] * 1.5, p0[1] * 1.5, p0[2] * 10, np.inf],
            ),
        )
    return fit


class GaussianFit(FrequencyDomain):

    def fit_func(self, f, f0, sigma_f, a, b):

        return a * np.exp(-((f - f0) ** 2) / (2 * sigma_f**2)) + b

    def _guess_init_params(self):
        """
        Guess initial parameters from data.
        """
        signal = self.signal
        f = self.frequency

        b0 = self._get_const_baseline()

        peak_a0, dip_a0 = np.max(signal) - b0, np.min(signal) - b0
        if peak_a0 > -dip_a0:  # peak detected case
            a0 = peak_a0
            f0 = f[np.argmax(signal)]
        else:  # valley detected case
            a0 = dip_a0
            f0 = f[np.argmin(signal)]

        # sigma linewidth is extracted from sample closest to 1/2 of max
        sigma_f0 = np.sqrt(np.log(2) / 2) * np.abs(
            f[np.argmin(np.abs(signal - (0.5 * a0 + b0)))] - f0
        )

        self.p0 = [f0, sigma_f0, a0, b0]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        self.f0 = popt[0]
        self.f0_sigma_err = np.sqrt(pcov[0, 0])
        self.sigma_f = popt[1]
        self.sigma_f_sigma_err = np.sqrt(pcov[1, 1])

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        # get most of the plotting done
        fig = self._plot_base()

        freq_fit = np.linspace(self.frequency[0], self.frequency[-1], fit_n_pts)

        plt.plot(
            freq_fit / self.frequency_scaler,
            self.fit_func(freq_fit, *(self.p0)),
            label="Fit (Init. Param.)",
            ls="--",
            lw=2,
            color="orange",
        )

        plt.plot(
            freq_fit / self.frequency_scaler,
            self.fit_func(freq_fit, *(self.popt)),
            label="Fit (Opt. Param.)",
            lw=2,
            color="red",
        )

        f0_string = (
            r"$f_0$ = %.4f $\pm$ %.4f "
            % (
                self.f0 / self.frequency_scaler,
                2 * self.f0_sigma_err / self.frequency_scaler,
            )
            + self.frequency_prefix
            + "Hz"
        )

        _, sigma_f_prefix = number_with_si_prefix(np.max(np.abs(self.sigma_f)))
        sigma_f_scaler = si_prefix_to_scaler(sigma_f_prefix)

        sigma_f_string = (
            r"$\sigma_f$ = %.4f $\pm$ %.4f "
            % (
                self.sigma_f / sigma_f_scaler,
                2 * self.sigma_f_sigma_err / sigma_f_scaler,
            )
            + sigma_f_prefix
            + "Hz"
        )

        plt.title(f0_string + ", " + sigma_f_string)
        plt.legend(loc=0, fontsize="x-small")

        fig.tight_layout()
        plt.show()

        return fig
