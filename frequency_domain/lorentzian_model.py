from base_freq_domain import *


def lorentzian_fit_func(f: float, f0: float, gamma: float, a: float, b: float) -> float:

    # angular frequency
    omega = 2 * np.pi * f
    omega0 = 2 * np.pi * f0

    return a + b / np.pi * (gamma / 2) / ((omega - omega0) ** 2 + (gamma / 2) ** 2)


def analyze_lorentzian(f: np.ndarray, sig: np.ndarray, p0: list[float] = None):
    sig_mag = sig
    if sig.dtype == complex:
        sig_mag = np.abs(sig) ** 2

    if p0 is None:
        if (np.max(sig_mag) - np.mean(sig_mag)) > (np.mean(sig_mag) - np.min(sig_mag)):
            # peak detected case
            f0 = f[np.argmax(sig_mag)]
            a = np.mean(
                sig_mag[np.argsort(sig_mag)[: int(len(sig_mag) // 10)]]
            )  # baseline (average of smallest 10% samples)
            # linewidth is extracted from sample closest to half-max
            gamma = 2 * np.abs(
                f[np.argmin(np.abs(sig_mag - 0.5 * (np.max(sig_mag) + a)))] - f0
            )
            b = np.pi * gamma / 2 * (np.max(sig_mag) - a)

            p0 = [f0, gamma, a, b]
        elif (np.max(sig_mag) - np.mean(sig_mag)) < (
            np.mean(sig_mag) - np.min(sig_mag)
        ):
            # valley detected case
            f0 = f[np.argmin(sig_mag)]
            a = np.mean(
                sig_mag[np.argsort(-sig_mag)[: int(len(sig) // 10)]]
            )  # baseline (average of largest 10% samples)
            # linewidth is extracted from sample closest to half-max
            gamma = 2 * np.abs(
                f[np.argmin(np.abs(sig_mag - 0.5 * (np.min(sig_mag) + a)))] - f0
            )
            b = np.pi * gamma / 2 * (np.min(sig_mag) - a)

            p0 = [f0, gamma, a, b]

    fit = curve_fit(
        lorentzian_fit_func,
        f,
        sig_mag,
        p0=p0,
        bounds=(
            [p0[0] * 0.5, p0[1] * 0.5, 0, p0[3] * 0.1],
            [p0[0] * 1.5, p0[1] * 1.5, np.inf, p0[3] * 10],
        ),
    )
    return fit


class LorentzianFit(FrequencyDomain):
    """ """

    def fit_func(self, f, f0, df, a, b):
        """
        Lorentzian fit function

        Parameters
        ----------
        f : TYPE
            DESCRIPTION.
        f0 : TYPE
            Resonant frequency.
        df : TYPE
            Full-width half-maximum linewidth.
        a : TYPE
            DESCRIPTION.
        b : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return a / ((f - f0) ** 2 + (df / 2) ** 2) + b

    def _guess_init_params(self):
        """
        Guess initial parameters from data.
        """
        signal = self.signal
        f = self.frequency

        b0 = self._get_const_baseline()

        peak_A0, dip_A0 = np.max(signal) - b0, np.min(signal) - b0
        if peak_A0 > -dip_A0:  # peak detected case
            A0 = peak_A0
            f0 = f[np.argmax(signal)]
        else:  # valley detected case
            A0 = dip_A0
            f0 = f[np.argmin(signal)]

        # linewidth is extracted from sample closest to half-max(arg1, arg2, _args)
        df0 = 2 * np.abs(f[np.argmin(np.abs(signal - (0.5 * A0 + b0)))] - f0)
        a0 = A0 * (df0 / 2) ** 2

        self.p0 = [f0, df0, a0, b0]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        self.f0 = popt[0]
        self.f0_sigma_err = np.sqrt(pcov[0, 0])
        self.df = popt[1]
        self.df_sigma_err = np.sqrt(pcov[1, 1])

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

        _, df_prefix = number_with_si_prefix(np.max(np.abs(self.df)))
        df_scaler = si_prefix_to_scaler(df_prefix)

        df_string = (
            r"$\Delta f$ = %.4f $\pm$ %.4f "
            % (self.df / df_scaler, 2 * self.df_sigma_err / df_scaler)
            + df_prefix
            + "Hz"
        )

        plt.title(f0_string + ", " + df_string)
        plt.legend(loc=0, fontsize="x-small")
        fig.tight_layout()
        plt.show()

        return fig
