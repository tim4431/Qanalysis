from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from Qanalysis.fitting.helper_tools import *
from base_freq_domain import FrequencyDomain


# class SingleSidedS11Fit(FrequencyDomain):
#     """
#     Class for implementing fitting of lineshape in reflection measurement
#     """
#     def __init__(self, freq, data, fit_type='complex'):
#         super().__init__(freq, data, df=df, fit_mag_dB=fit_mag_dB, plot_mag_dB=plot_mag_dB)

#         self.fit_type = fit_type

#     def fit_func(self, *args):
#         """
#         Reflection fit for resonator single-sided coupled to waveguide
#         according to relation
#         S11 = a * exp(i(ϕ + 2pi * (f - f[0]) * τ)) .* ...
#              (1 - 2 * QoverQe * (1 + 2i * δf / f0) / (1 + 2i * Q * (f - f0) / f0))

#         Note: If df = 0 (no background) this gives
#         S11 = 1 - (2 kappa_e) / (kappa_i + kappa_e + 2i (omega - omega0))
#             = (kappa_i - kappa_e + 2i (omega - omega0)) / (kappa_i + kappa_e + 2i (omega - omega0))
#         See Aspelmeyer et al, "Cavity Optomechanics", Rev. Mod. Phys. 86, 1391 (2014).

#         If the fit type is complex, the arguments of `fit_func` are:
#             f, f0, Q, QoverQe, df, a, phi, tau
#         If the fit type is magnitude, the arguments of `fit_func` are:
#             f, f0, Q, QoverQe, df, a
#         If the fit type is phase, the arguments of `fit_func` are:
#             f, f0, Q, QoverQe, df, phi, tau
#         """
#         if self.fit_type == 'complex':
#             f, f0, Q, QoverQe, df, a, phi, tau = args

#             return (a * np.exp(1j * (phi + 2 * np.pi * (f - f[0]) * tau)) *
#                     (1 - 2 * QoverQe * (1 + 2j * df / f0) /
#                     (1 + 2j * Q * (f - f0) / f0)))
#         elif self.fit_type == 'magnitude':
#             f, f0, Q, QoverQe, df, a = args

#             return np.abs(a * (1 - 2 * QoverQe * (1 + 2j * df / f0) /
#                                (1 + 2j * Q * (f - f0) / f0)))
#         elif self.fit_type == 'phase':
#             f, f0, Q, QoverQe, df, phi, tau = args

#             return np.angle(np.exp(1j * (phi + 2 * np.pi * (f - f[0]) * tau)) *
#                             (1 - 2 * QoverQe * (1 + 2j * df / f0) /
#                              (1 + 2j * Q * (f - f0) / f0)))

#     def _estimate_f0_FWHM(self):
#         f = self.frequency
#         mag2 = np.abs(self.data) ** 2

#         # background in magnitude estimated by linear interpolation of first and last point
#         mag2_bg = (mag2[-1] - mag2[0]) / (f[-1] - f[0]) * (f - f[0]) + mag2[0]

#         mag2_subtracted = mag2 - mag2_bg
#         f0 = f[mag2_subtracted.argmin()]
#         smin, smax = np.min(mag2_subtracted), np.max(mag2_subtracted)

#         # data in frequency < f0 or frequency >= f0
#         f_l, s_l = f[f < f0], mag2_subtracted[f < f0]
#         f_r, s_r = f[f >= f0], mag2_subtracted[f >= f0]

#         # find frequencies closest to the mean of smin and smax
#         f1 = f_l[np.abs(s_l - 0.5 * (smin + smax)).argmin()]
#         f2 = f_r[np.abs(s_r - 0.5 * (smin + smax)).argmin()]

#         # numerically find full width half max from magnitude squared
#         Δf = f2 - f1

#         return f0, Δf

#     def _guess_init_params(self, df):

#         # magnitude data
#         _mag = np.abs(self.data)
#         # phase data
#         _ang = np.angle(self.data)
#         # unwrapped phase data
#         _angU = np.unwrap(_ang)

#         f = self.frequency

#         a0 = np.sqrt((_mag[0] ** 2 + _mag[-1] ** 2) / 2)
#         phi0 = _angU[0]
#         tau0 = 0.0
#         if (np.max(_angU) - np.min(_angU)) > 2.1 * np.pi:
#             # if phase data at start and stop frequencies differ by more than 2pi,
#             # perform phase subtraction associated with delay
#             tau0 = (_angU[-1] - _angU[0]) / ((f[-1] - f[0]))/ (2 * np.pi)

#         # Estimate total Q from the FWHM in |mag|^2
#         f0, Δf = self._estimate_f0_FWHM()
#         amin = _mag[np.abs(f - f0).argmin()]
#         QoverQe0 = 0.5 * (1 - amin / a0)
#         Q0 = f0 / Δf

#         if self.fit_type == 'complex':
#             self.p0 = [f0, Q0, QoverQe0, 0.0, a0, phi0, tau0]
#         elif self.fit_type == 'magnitude':
#             self.p0 = [f0, Q0, QoverQe0, 0.0, a0]
#         elif self.fit_type == 'phase':
#             self.p0 = [f0, Q0, QoverQe0, 0.0, phi0, tau0]


# class acStarkShift:
#     """
#     Class to implement analysis of ac Stark shift measurement with varying
#     power applied to the readout resonator.

#     Parameters
#     ----------
#     freq : `numpy.ndarray`
#         A 1D numpy array that specifies the range of frequencies (in units of
#         Hz) for qubit spectroscopy.
#     power_dBm : `numpy.ndarray`
#         A 1D numpy array that specifies the range of power (in units of dBm)
#         for readout resonator.
#     signal : `numpy.ndarray`
#         A 2D numpy array that stores the complex signal obtained from qubit
#         spectroscopy with taken with a range of readout power. The row indices
#         and column indices correspond to indices of `power_dBm` and `freq`,
#         respectively.
#     disp_freq_shift : `float`
#         Dispersive frequency shift of readout resonator 2 * chi / (2 * pi) in
#         units of Hz.

#     Attributes
#     ----------


#     Methods
#     -------

#     """
#     def __init__(self, freq, power_dBm, signal, disp_freq_shift):
#         self.frequency = freq
#         # complex signal obtained from spectroscopy
#         self.signal = signal
#         # readout power in units of dBm
#         self.power_dBm = power_dBm
#         # dispersive frequency shift (2*chi / 2*pi) in units of Hz
#         self.disp_freq_shift = disp_freq_shift # 2 * chi / (2 * pi)

#         self.p0 = None
#         self.popt = None
#         self.pcov = None

#     def analyze(self, plot=True, p0=None):

#         self.res_frequency = np.zeros(len(self.power_dBm))
#         for idx in range(len(self.power_dBm)):
#             lorentzian_fit = analyze_lorentzian(self.frequency, self.signal[idx])
#             self.res_frequency[idx] = lorentzian_fit[0][0]

#         self._set_init_params(p0)

#         self.popt, self.pcov = curve_fit(self.fit_func, self.power_dBm,
#                                          self.res_frequency, p0=self.p0)
#         a, f0 = self.popt

#         self.single_photon_power_dBm = watt_to_dBm(self.disp_freq_shift / a)

#         if plot:
#             self.plot_result()

#     def _set_init_params(self, p0):
#         if p0 is None:
#             dp = dBm_to_watt(np.max(self.power_dBm))
#             maxind, minind = np.argmax(self.power_dBm), np.argmin(self.power_dBm)
#             df = self.res_frequency[maxind] - self.res_frequency[minind]
#             self.p0 = [df / dp, self.res_frequency[np.argmin(self.power_dBm)]]
#         else:
#             self.p0 = p0

#     def fit_func(self, power_dBm, a, f0):

#         power_Watt = dBm_to_watt(power_dBm)
#         return a * power_Watt + f0

#     def plot_result(self):
#         fig = plt.figure()
#         plt.pcolormesh(self.frequency / 1e9, self.power_dBm, np.abs(self.signal), shading='auto')
#         plt.plot(self.res_frequency / 1e9, self.power_dBm, '.', color='k', label='Res. Freq.')
#         plt.plot(self.fit_func(self.power_dBm, *self.popt) / 1e9,
#                  self.power_dBm, '--', color='white', label='Fit')
#         plt.title(r"Single-photon power = $%.2f$ dBm" % self.single_photon_power_dBm)
#         plt.legend()
#         plt.xlabel("Frequency (GHz)")
#         plt.ylabel("Power (dBm)")


# class WaveguideCoupledS21Fit(FrequencyDomain):
#     """
#     Class for imlementing fitting of lineshape in reflection measurement
#     """
#     def __init__(self, freq, data, df=0, fit_mag_dB=False, plot_mag_dB=False):
#         super().__init__(freq, data, df=df, fit_mag_dB=fit_mag_dB, plot_mag_dB=plot_mag_dB)

#         self.fit_type = "WaveguideCoupledTransmission"

#     def fit_func(self, f, f0_MHz, Q, QoverQe, δf, a, ϕ, τ_ns):
#         """
#         Reflection fit for resonator single-sided coupled to waveguide
#         according to relation
#         S11 = a0 * exp(i(ϕ0 + 2pi * (f - f[0]) * τ0)) .* ...
#              (1 - QoverQe * (1 + 2i * δf / f0) / (1 + 2i * Q * (f - f0) / f0))

#         Note: If df = 0 (no background) this gives
#         S21 = 1 - (kappa_e) / (kappa_i + kappa_e + 2i (omega - omega0))
#             = (kappa_i + 2i (omega - omega0)) / (kappa_i + kappa_e + 2i (omega - omega0))
#         See Khalil et al, "An analysis method for asymmetric resonator
#         transmission applied to superconducting devices",
#         J. Appl. Phys. 111, 054510 (2012).
#         """
#         return (a * np.exp(1j * (ϕ + 2 * np.pi * (f - f[0]) * τ_ns * 1e-9)) *
#                 (1 - QoverQe * (1 + 2j * δf / f0_MHz) /
#                 (1 + 2j * (Q) * (f/1e6 - f0_MHz) / (f0_MHz))))

#     def _init_fit_params(self, df):
#         # magnitude data
#         _mag = np.abs(self.data)
#         # phase data
#         _ang = np.angle(self.data)
#         # unwrapped phase data
#         _angU = np.unwrap(_ang)

#         f = self.frequency

#         a0 = _mag[0]
#         ϕ0 = _angU[0]
#         τ0 = 0.0
#         if (np.max(_angU) - np.min(_angU)) > 2.1 * np.pi:
#             # if phase data at start and stop frequencies differ more than 2pi,
#             # perform phase subtraction associated with delay
#             τ0 = (_angU[-1] - _angU[0]) / ((f[-1] - f[0]))/ (2 * np.pi)

#         # Estimate total Q from the FWHM in |mag|^2
#         f0, Δf = self._estimate_f0_FWHM()
#         QoverQe0 = (1 - np.min(_mag) / a0)
#         Q0 = f0 / Δf
#         p0_mag, p0_ang = self._prepare_fit_params(f0, Q0, QoverQe0,
#                                                   df, a0, ϕ0, τ0)

#         self.p0 = p0_mag + p0_ang
#         return p0_mag, p0_ang
