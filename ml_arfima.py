import pandas as pd
import numpy as np

class ML_ARFIMA:
    
    def __init__(self, df=None):
        import warnings
        warnings.filterwarnings("ignore")

        self.df = df
        self.d = 0
        
    def data(self, df):

        self.df = pd.DataFrame(df)
        return df
    
    def tsplot(self, lags=None, title='', figsize=(14, 8)):

        '''Examine the patterns of ACF and PACF, along with the time series plot and histogram.
        
        Original source: https://tomaugspurger.github.io/modern-7-timeseries.html
        '''
        
        from matplotlib import pyplot as plt
        import statsmodels.tsa.api as smt
        import seaborn as sns
        
        if type(self.df) == pd.DataFrame:
            fig = plt.figure(figsize=figsize)
            layout = (2, 2)
            ts_ax   = plt.subplot2grid(layout, (0, 0))
            hist_ax = plt.subplot2grid(layout, (0, 1))
            acf_ax  = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
        
            self.df.plot(ax=ts_ax)
            ts_ax.set_title(title)
            self.df.plot(ax=hist_ax, kind='hist', bins=25)
            hist_ax.set_title('Histogram')
            smt.graphics.plot_acf(self.df, lags=lags, ax=acf_ax)
            smt.graphics.plot_pacf(self.df, lags=lags, ax=pacf_ax)
            [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
            sns.despine()
            plt.tight_layout()

            return ts_ax, acf_ax, pacf_ax
        else:
            print('No Data!')
    
    def adf(self):
        import statsmodels.tsa.stattools as st
        
        if type(self.df) == pd.DataFrame:
            adf_result = st.adfuller(self.df, store = True)
            print("ADF Test Results: ")
            print("Test Statistic: %.4f" % adf_result[0])
            print("p-value: %.10f" % adf_result[1])
        else:
            print('No Data!')
        
    def kpss(self):
        import statsmodels.tsa.stattools as st
        
        if type(self.df) == pd.DataFrame:
            kpss_result = st.kpss(self.df, store=True)
            print("KPSS Test Results: ")
            print("Test Statistic: %.4f" % kpss_result[0])
            print("p-value: %.10f" % kpss_result[1])
        else:
            print('No Data!')
            
    def AIC_BIC_HIQC(self, max_ar = 4, max_ma= 4, trend='n'):
        import statsmodels.api as sm
        
        if type(self.df) == pd.DataFrame:
            train_results = sm.tsa.arma_order_select_ic(self.df, ic=['aic', 'bic', 'hqic'], trend=trend, 
                                                        max_ar=max_ar, max_ma=max_ma)
            self.aic = train_results.aic_min_order
            self.bic = train_results.bic_min_order
            self.hqic = train_results.hqic_min_order
            print('AIC', self.aic)
            print('BIC', self.bic)
            print('HQIC', self.hqic)
        else:
            print('No Data!')
            
    def diff_(self, d):
        from scipy.fftpack import fft, ifft
        
        """Fast fractional difference algorithm (by Jensen & Nielsen (2014)).
        """
        
        list_ = [i[0] for i in self.df.values]

        def next_pow2(n):
            # we assume that the input will always be n > 1,
            # so this brief calculation should be fine
            return (n - 1).bit_length()

        n_points = len(list_)
        fft_len = 2 ** next_pow2(2 * n_points - 1)
        prod_ids = np.arange(1, n_points)
        frac_diff_coefs = np.append([1], np.cumprod((prod_ids - d - 1) / prod_ids))
        dx = ifft(fft(list_, fft_len) * fft(frac_diff_coefs, fft_len))

        self.d = self.d + d
        self.df = pd.DataFrame(np.real(dx[0:n_points]), index=self.df.index, columns=self.df.columns)
        return self.df
    
    def fit(self, order):
        from statsmodels.tsa.arima.model import ARIMA
        
        model = ARIMA(self.df, order = (order[0], 0, order[1]))
        model_fit = model.fit()
        self.model = model_fit
        return model_fit.summary()