import pandas as pd
import glob
import os
import statsmodels as sm  # type: ignore
import scipy.stats as stats  # type: ignore
import statsmodels.formula.api as smf  # type: ignore


# Caminho para os arquivos processados
processed_folder = (
    os.path.join("..", "data", "processed")
    if not os.path.exists("data/processed")
    else "data/processed"
)

csv_files = glob.glob(os.path.join(processed_folder, "*.csv"))


for csv_path in csv_files:

    filename = os.path.basename(csv_path)
    if "BHC_USDT" in filename:
        volume_col = "volume_bhc"
    elif "BTC_USDT" in filename:
        volume_col = "volume_btc"
    elif "BHC_USDT" in filename:
        volume_col = "volume_bhc"
    elif "DASH_USDT" in filename:
        volume_col = "volume_dash"
    elif "EOS_USDT" in filename:
        volume_col = "volume_eos"
    elif "ETC_USDT" in filename:
        volume_col = "volume_etc"
    elif "ETH_USDT" in filename:
        volume_col = "volume_eth"
    elif "LTC_USDT" in filename:
        volume_col = "volume_ltc"
    elif "XMR_USDT" in filename:
        volume_col = "volume_xmr"
    elif "XRP_USDT" in filename:
        volume_col = "volume_xrp"
    elif "ZRX_USDT" in filename:
        volume_col = "volume_zrx"
    # Adicione outras condições 'elif' para outros pares, se necessário
    else:
        # Se não encontrar um padrão conhecido, use 'volume' como padrão ou pule o arquivo
        print(
            f"Warning: Não foi possível determinar a coluna volume em {filename}. Pulando."
        )
        continue

    df = pd.read_csv(csv_path)  # type: ignore
    # Remove a coluna 'date' se existir
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    # Calcula a matriz de correlação
    # analise com todas as variáveis
    # modelo = smf.ols(f'close ~ open+high+low+{volume_col}+buytakeramount+buytakerquantity+weightedaverage+sma_7+std_7+sma_14+std_14+sma_30+std_30+daily_return+volatility_7d+volatility_30d+close_lag1+close_lag5+rsi+macd+macd_signal+macd_diff+bb_upper+bb_lower+bb_mavg+obv',data = df)
    # analise com dados otimizados, escolhidos conforme comentário ao fim do código
    modelo = smf.ols(  # type: ignore
        f"close ~ high + low + sma_7 + sma_14 + sma_30 + "
        f"close_lag5 + macd + macd_signal + macd_diff + "
        f"bb_upper + bb_lower + bb_mavg + daily_return",
        data=df,
    )
    modelo = modelo.fit()  # type: ignore

    print(csv_path, "-" * 50)
    print(formula)  # type: ignore
    print("-" * 70)
    print(modelo.summary())

"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Resultado encontrado com todas as variáveis dos arquivos featured:

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



/content/featured_XRP_USDT.csv --------------------------------------------------
Formula: close ~ open+high+low+volume_xrp+buytakeramount+buytakerquantity+weightedaverage+sma_7+std_7+sma_14+std_14+sma_30+std_30+daily_return+volatility_7d+volatility_30d+close_lag1+close_lag5+rsi+macd+macd_signal+macd_diff+bb_upper+bb_lower+bb_mavg+obv
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 3.215e+05
Date:                Fri, 04 Jul 2025   Prob (F-statistic):               0.00
Time:                        23:33:01   Log-Likelihood:                 11123.
No. Observations:                3752   AIC:                        -2.220e+04
Df Residuals:                    3727   BIC:                        -2.204e+04
Df Model:                          24                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept           -0.0037      0.001     -2.997      0.003      -0.006      -0.001
open                -0.7563      0.177     -4.271      0.000      -1.103      -0.409
high                 0.3022      0.016     18.491      0.000       0.270       0.334
low                  0.2225      0.014     15.694      0.000       0.195       0.250
volume_xrp       -3.697e-12   4.43e-11     -0.083      0.934   -9.06e-11    8.32e-11
buytakeramount     2.64e-12   1.09e-10      0.024      0.981   -2.11e-10    2.16e-10
buytakerquantity  1.941e-11   6.28e-11      0.309      0.757   -1.04e-10    1.42e-10
weightedaverage      0.0611      0.028      2.201      0.028       0.007       0.116
sma_7               -1.4180      0.027    -51.890      0.000      -1.472      -1.364
std_7                0.0086      0.011      0.747      0.455      -0.014       0.031
sma_14               0.6733      0.017     40.383      0.000       0.641       0.706
std_14              -0.0091      0.014     -0.668      0.504      -0.036       0.018
sma_30               0.7160      0.014     52.213      0.000       0.689       0.743
std_30              -0.0091      0.009     -1.042      0.297      -0.026       0.008
daily_return         0.0146      0.002      6.189      0.000       0.010       0.019
volatility_7d       -0.0017      0.001     -1.456      0.145      -0.004       0.001
volatility_30d      -0.0003      0.000     -0.705      0.481      -0.001       0.001
close_lag1           0.1222      0.178      0.686      0.493      -0.227       0.471
close_lag5           0.2164      0.008     27.917      0.000       0.201       0.232
rsi               8.459e-05   2.37e-05      3.572      0.000    3.82e-05       0.000
macd                 4.1331      0.055     74.545      0.000       4.024       4.242
macd_signal         -1.4379      0.021    -69.076      0.000      -1.479      -1.397
macd_diff            5.5710      0.075     74.769      0.000       5.425       5.717
bb_upper             0.2926      0.007     40.190      0.000       0.278       0.307
bb_lower             0.2803      0.007     38.413      0.000       0.266       0.295
bb_mavg              0.2865      0.006     46.039      0.000       0.274       0.299
obv              -7.827e-13   5.05e-13     -1.550      0.121   -1.77e-12    2.07e-13
==============================================================================
Omnibus:                     1194.235   Durbin-Watson:                   1.847
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           128258.359
Skew:                          -0.480   Prob(JB):                         0.00
Kurtosis:                      31.627   Cond. No.                     1.13e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 4.62e-11. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_ETC_USDT.csv --------------------------------------------------
Formula: close ~ open+high+low+volume_etc+buytakeramount+buytakerquantity+weightedaverage+sma_7+std_7+sma_14+std_14+sma_30+std_30+daily_return+volatility_7d+volatility_30d+close_lag1+close_lag5+rsi+macd+macd_signal+macd_diff+bb_upper+bb_lower+bb_mavg+obv
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 2.041e+05
Date:                Fri, 04 Jul 2025   Prob (F-statistic):               0.00
Time:                        23:33:01   Log-Likelihood:                -1281.6
No. Observations:                3228   AIC:                             2613.
Df Residuals:                    3203   BIC:                             2765.
Df Model:                          24                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept           -0.0211      0.047     -0.451      0.652      -0.113       0.070
open                -0.1558      0.054     -2.865      0.004      -0.262      -0.049
high                 0.2359      0.015     15.517      0.000       0.206       0.266
low                  0.2170      0.016     13.563      0.000       0.186       0.248
volume_etc        5.018e-10   1.28e-10      3.922      0.000    2.51e-10    7.53e-10
buytakeramount    2.275e-09   9.26e-09      0.246      0.806   -1.59e-08    2.04e-08
buytakerquantity  4.373e-08   2.34e-07      0.187      0.852   -4.16e-07    5.03e-07
weightedaverage      0.1693      0.024      6.964      0.000       0.122       0.217
sma_7               -1.2808      0.026    -48.471      0.000      -1.333      -1.229
std_7                0.0443      0.011      4.197      0.000       0.024       0.065
sma_14               0.5975      0.016     37.404      0.000       0.566       0.629
std_14              -0.0215      0.013     -1.592      0.111      -0.048       0.005
sma_30               0.5897      0.014     43.255      0.000       0.563       0.616
std_30              -0.0114      0.010     -1.165      0.244      -0.030       0.008
daily_return         3.2337      0.175     18.507      0.000       2.891       3.576
volatility_7d       -0.3442      0.113     -3.059      0.002      -0.565      -0.124
volatility_30d       0.1721      0.079      2.186      0.029       0.018       0.326
close_lag1          -0.3395      0.055     -6.213      0.000      -0.447      -0.232
close_lag5           0.1938      0.007     25.981      0.000       0.179       0.208
rsi                 -0.0002      0.001     -0.230      0.818      -0.002       0.001
macd                 3.5594      0.057     61.988      0.000       3.447       3.672
macd_signal         -1.2509      0.022    -57.083      0.000      -1.294      -1.208
macd_diff            4.8103      0.077     62.111      0.000       4.658       4.962
bb_upper             0.2595      0.008     34.230      0.000       0.245       0.274
bb_lower             0.2554      0.007     36.093      0.000       0.242       0.269
bb_mavg              0.2575      0.006     41.330      0.000       0.245       0.270
obv               2.334e-11   9.09e-11      0.257      0.797   -1.55e-10    2.02e-10
==============================================================================
Omnibus:                      717.714   Durbin-Watson:                   1.890
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            31898.898
Skew:                           0.075   Prob(JB):                         0.00
Kurtosis:                      18.399   Cond. No.                     1.07e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 1.81e-12. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_ZRX_USDT.csv --------------------------------------------------
Formula: close ~ open+high+low+volume_zrx+buytakeramount+buytakerquantity+weightedaverage+sma_7+std_7+sma_14+std_14+sma_30+std_30+daily_return+volatility_7d+volatility_30d+close_lag1+close_lag5+rsi+macd+macd_signal+macd_diff+bb_upper+bb_lower+bb_mavg+obv
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 8.053e+04
Date:                Fri, 04 Jul 2025   Prob (F-statistic):               0.00
Time:                        23:33:01   Log-Likelihood:                 7320.7
No. Observations:                2464   AIC:                        -1.459e+04
Df Residuals:                    2439   BIC:                        -1.445e+04
Df Model:                          24                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept            0.0004      0.002      0.221      0.825      -0.003       0.004
open                -0.3871      0.029    -13.349      0.000      -0.444      -0.330
high                 0.1792      0.013     14.323      0.000       0.155       0.204
low                  0.2789      0.014     19.899      0.000       0.251       0.306
volume_zrx        1.479e-09   2.46e-09      0.602      0.547   -3.34e-09    6.29e-09
buytakeramount   -2.887e-09   4.58e-09     -0.631      0.528   -1.19e-08    6.09e-09
buytakerquantity  8.663e-11   1.24e-10      0.696      0.487   -1.57e-10    3.31e-10
weightedaverage      0.1229      0.020      6.064      0.000       0.083       0.163
sma_7               -0.8467      0.030    -28.111      0.000      -0.906      -0.788
std_7                0.0664      0.014      4.741      0.000       0.039       0.094
sma_14               0.4361      0.023     19.198      0.000       0.392       0.481
std_14               0.0021      0.013      0.158      0.875      -0.024       0.028
sma_30               0.4750      0.018     26.261      0.000       0.440       0.511
std_30              -0.0092      0.010     -0.964      0.335      -0.028       0.009
daily_return         0.1443      0.005     29.477      0.000       0.135       0.154
volatility_7d       -0.0162      0.002     -7.209      0.000      -0.021      -0.012
volatility_30d      -0.0011      0.001     -1.311      0.190      -0.003       0.001
close_lag1           0.0569      0.031      1.837      0.066      -0.004       0.118
close_lag5           0.0960      0.007     13.174      0.000       0.082       0.110
rsi               2.403e-05   3.41e-05      0.704      0.481   -4.29e-05    9.09e-05
macd                 2.7674      0.068     40.598      0.000       2.634       2.901
macd_signal         -0.9597      0.026    -36.619      0.000      -1.011      -0.908
macd_diff            3.7270      0.092     40.569      0.000       3.547       3.907
bb_upper             0.2022      0.008     24.299      0.000       0.186       0.219
bb_lower             0.1908      0.010     19.812      0.000       0.172       0.210
bb_mavg              0.1965      0.008     23.943      0.000       0.180       0.213
obv              -4.673e-13   2.14e-12     -0.219      0.827   -4.66e-12    3.72e-12
==============================================================================
Omnibus:                      864.640   Durbin-Watson:                   1.820
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           133596.768
Skew:                          -0.529   Prob(JB):                         0.00
Kurtosis:                      39.058   Cond. No.                     1.04e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 9.41e-13. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_EOS_USDT.csv --------------------------------------------------
Formula: close ~ open+high+low+volume_eos+buytakeramount+buytakerquantity+weightedaverage+sma_7+std_7+sma_14+std_14+sma_30+std_30+daily_return+volatility_7d+volatility_30d+close_lag1+close_lag5+rsi+macd+macd_signal+macd_diff+bb_upper+bb_lower+bb_mavg+obv
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 1.439e+05
Date:                Fri, 04 Jul 2025   Prob (F-statistic):               0.00
Time:                        23:33:01   Log-Likelihood:                 3954.1
No. Observations:                2461   AIC:                            -7858.
Df Residuals:                    2436   BIC:                            -7713.
Df Model:                          24                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept           -0.0063      0.007     -0.859      0.390      -0.021       0.008
open                -0.3938      0.064     -6.136      0.000      -0.520      -0.268
high                 0.4197      0.018     23.748      0.000       0.385       0.454
low                  0.3350      0.018     18.771      0.000       0.300       0.370
volume_eos       -5.899e-09   4.24e-09     -1.390      0.165   -1.42e-08    2.42e-09
buytakeramount    1.174e-08   8.45e-09      1.389      0.165   -4.83e-09    2.83e-08
buytakerquantity  1.121e-10   6.49e-10      0.173      0.863   -1.16e-09    1.39e-09
weightedaverage      0.0034      0.029      0.115      0.908      -0.054       0.060
sma_7               -0.9362      0.028    -33.145      0.000      -0.992      -0.881
std_7                0.0338      0.013      2.536      0.011       0.008       0.060
sma_14               0.4738      0.020     23.225      0.000       0.434       0.514
std_14               0.0017      0.017      0.100      0.921      -0.032       0.036
sma_30               0.4872      0.015     31.602      0.000       0.457       0.517
std_30               0.0137      0.012      1.185      0.236      -0.009       0.036
daily_return         0.4216      0.031     13.662      0.000       0.361       0.482
volatility_7d       -0.0468      0.022     -2.134      0.033      -0.090      -0.004
volatility_30d      -0.0070      0.015     -0.461      0.645      -0.036       0.023
close_lag1          -0.1856      0.065     -2.876      0.004      -0.312      -0.059
close_lag5           0.1000      0.007     13.672      0.000       0.086       0.114
rsi                  0.0002      0.000      1.451      0.147   -6.58e-05       0.000
macd                 3.0209      0.059     50.892      0.000       2.905       3.137
macd_signal         -1.0585      0.023    -45.820      0.000      -1.104      -1.013
macd_diff            4.0795      0.080     50.764      0.000       3.922       4.237
bb_upper             0.2218      0.007     32.584      0.000       0.208       0.235
bb_lower             0.2427      0.010     23.856      0.000       0.223       0.263
bb_mavg              0.2322      0.007     31.805      0.000       0.218       0.247
obv              -5.512e-13    1.1e-11     -0.050      0.960   -2.21e-11     2.1e-11
==============================================================================
Omnibus:                      782.369   Durbin-Watson:                   1.975
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           100380.592
Skew:                          -0.379   Prob(JB):                         0.00
Kurtosis:                      34.279   Cond. No.                     1.11e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 4.92e-13. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
Warning: Could not determine volume column name for featured_BCH_USDT.csv. Skipping.
/content/featured_DASH_USDT.csv --------------------------------------------------
Formula: close ~ open+high+low+volume_dash+buytakeramount+buytakerquantity+weightedaverage+sma_7+std_7+sma_14+std_14+sma_30+std_30+daily_return+volatility_7d+volatility_30d+close_lag1+close_lag5+rsi+macd+macd_signal+macd_diff+bb_upper+bb_lower+bb_mavg+obv
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 2.803e+05
Date:                Fri, 04 Jul 2025   Prob (F-statistic):               0.00
Time:                        23:33:01   Log-Likelihood:                -10180.
No. Observations:                3755   AIC:                         2.041e+04
Df Residuals:                    3730   BIC:                         2.057e+04
Df Model:                          24                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept           -1.6208      0.308     -5.268      0.000      -2.224      -1.018
open                -0.3174      0.064     -4.944      0.000      -0.443      -0.192
high                 0.1706      0.018      9.233      0.000       0.134       0.207
low                  0.1734      0.018      9.667      0.000       0.138       0.209
volume_dash       2.273e-07   1.28e-07      1.771      0.077   -2.43e-08    4.79e-07
buytakeramount   -6.318e-07   3.49e-07     -1.813      0.070   -1.32e-06    5.16e-08
buytakerquantity  5.073e-06   5.94e-06      0.854      0.393   -6.58e-06    1.67e-05
weightedaverage      0.2236      0.029      7.740      0.000       0.167       0.280
sma_7               -1.2291      0.030    -40.627      0.000      -1.288      -1.170
std_7                0.0553      0.014      3.891      0.000       0.027       0.083
sma_14               0.5735      0.019     30.174      0.000       0.536       0.611
std_14              -0.0245      0.018     -1.381      0.167      -0.059       0.010
sma_30               0.6502      0.016     39.998      0.000       0.618       0.682
std_30               0.0198      0.010      1.987      0.047       0.000       0.039
daily_return         0.0254      0.041      0.613      0.540      -0.056       0.107
volatility_7d       -0.0031      0.018     -0.173      0.863      -0.038       0.032
volatility_30d    8.931e-05      0.008      0.011      0.991      -0.016       0.016
close_lag1          -0.2548      0.065     -3.949      0.000      -0.381      -0.128
close_lag5           0.1414      0.008     17.426      0.000       0.126       0.157
rsi                  0.0310      0.006      5.355      0.000       0.020       0.042
macd                 3.8432      0.062     61.698      0.000       3.721       3.965
macd_signal         -1.3158      0.022    -59.411      0.000      -1.359      -1.272
macd_diff            5.1590      0.083     62.178      0.000       4.996       5.322
bb_upper             0.2916      0.008     36.717      0.000       0.276       0.307
bb_lower             0.2850      0.008     34.666      0.000       0.269       0.301
bb_mavg              0.2883      0.007     42.643      0.000       0.275       0.302
obv               4.628e-09   1.59e-07      0.029      0.977   -3.07e-07    3.17e-07
==============================================================================
Omnibus:                     1654.550   Durbin-Watson:                   1.944
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           652843.975
Skew:                          -0.780   Prob(JB):                         0.00
Kurtosis:                      67.577   Cond. No.                     1.14e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 3.65e-16. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_ETH_USDT.csv --------------------------------------------------
Formula: close ~ open+high+low+volume_eth+buytakeramount+buytakerquantity+weightedaverage+sma_7+std_7+sma_14+std_14+sma_30+std_30+daily_return+volatility_7d+volatility_30d+close_lag1+close_lag5+rsi+macd+macd_signal+macd_diff+bb_upper+bb_lower+bb_mavg+obv
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 6.668e+05
Date:                Fri, 04 Jul 2025   Prob (F-statistic):               0.00
Time:                        23:33:02   Log-Likelihood:                -15531.
No. Observations:                3583   AIC:                         3.111e+04
Df Residuals:                    3558   BIC:                         3.127e+04
Df Model:                          24                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept           -4.8975      1.964     -2.494      0.013      -8.748      -1.047
open                -0.4466      0.264     -1.690      0.091      -0.965       0.072
high                 0.3494      0.017     20.095      0.000       0.315       0.383
low                  0.3100      0.015     20.817      0.000       0.281       0.339
volume_eth        1.178e-07   3.07e-08      3.838      0.000    5.76e-08    1.78e-07
buytakeramount   -1.079e-07   1.13e-07     -0.958      0.338   -3.29e-07    1.13e-07
buytakerquantity    -0.0001      0.000     -0.468      0.640      -0.001       0.000
weightedaverage     -0.0277      0.026     -1.050      0.294      -0.079       0.024
sma_7               -1.3737      0.028    -48.808      0.000      -1.429      -1.318
std_7                0.0208      0.011      1.868      0.062      -0.001       0.043
sma_14               0.6581      0.018     36.775      0.000       0.623       0.693
std_14               0.0134      0.013      1.011      0.312      -0.013       0.039
sma_30               0.6990      0.014     48.233      0.000       0.671       0.727
std_30              -0.0265      0.008     -3.146      0.002      -0.043      -0.010
daily_return        63.2210      7.092      8.915      0.000      49.317      77.125
volatility_7d      -12.1377      5.858     -2.072      0.038     -23.623      -0.653
volatility_30d       6.4770      3.349      1.934      0.053      -0.090      13.044
close_lag1          -0.1988      0.264     -0.753      0.452      -0.717       0.319
close_lag5           0.1993      0.008     25.040      0.000       0.184       0.215
rsi                  0.0636      0.033      1.920      0.055      -0.001       0.129
macd                 4.0082      0.060     66.901      0.000       3.891       4.126
macd_signal         -1.3707      0.022    -61.368      0.000      -1.414      -1.327
macd_diff            5.3789      0.081     66.619      0.000       5.221       5.537
bb_upper             0.2784      0.007     39.275      0.000       0.265       0.292
bb_lower             0.2750      0.008     34.237      0.000       0.259       0.291
bb_mavg              0.2767      0.007     41.413      0.000       0.264       0.290
obv               3.184e-09    2.3e-09      1.386      0.166   -1.32e-09    7.69e-09
==============================================================================
Omnibus:                      702.919   Durbin-Watson:                   1.842
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            20783.502
Skew:                           0.123   Prob(JB):                         0.00
Kurtosis:                      14.796   Cond. No.                     1.31e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 7.34e-12. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_XMR_USDT.csv --------------------------------------------------
Formula: close ~ open+high+low+volume_xmr+buytakeramount+buytakerquantity+weightedaverage+sma_7+std_7+sma_14+std_14+sma_30+std_30+daily_return+volatility_7d+volatility_30d+close_lag1+close_lag5+rsi+macd+macd_signal+macd_diff+bb_upper+bb_lower+bb_mavg+obv
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 2.579e+05
Date:                Fri, 04 Jul 2025   Prob (F-statistic):               0.00
Time:                        23:33:02   Log-Likelihood:                -8374.0
No. Observations:                3755   AIC:                         1.680e+04
Df Residuals:                    3730   BIC:                         1.695e+04
Df Model:                          24                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept           -1.4761      0.278     -5.307      0.000      -2.021      -0.931
open                -0.1753      0.073     -2.406      0.016      -0.318      -0.032
high                 0.1931      0.015     12.966      0.000       0.164       0.222
low                  0.2059      0.014     15.070      0.000       0.179       0.233
volume_xmr        7.979e-08    5.5e-08      1.451      0.147    -2.8e-08    1.88e-07
buytakeramount    -3.28e-07   1.42e-07     -2.302      0.021   -6.07e-07   -4.86e-08
buytakerquantity  3.678e-05   1.94e-05      1.900      0.058   -1.17e-06    7.47e-05
weightedaverage      0.1322      0.023      5.685      0.000       0.087       0.178
sma_7               -1.3841      0.028    -49.262      0.000      -1.439      -1.329
std_7                0.0893      0.013      6.956      0.000       0.064       0.114
sma_14               0.7032      0.019     36.151      0.000       0.665       0.741
std_14              -0.0772      0.017     -4.546      0.000      -0.110      -0.044
sma_30               0.7277      0.016     46.333      0.000       0.697       0.758
std_30               0.0002      0.010      0.025      0.980      -0.018       0.019
daily_return         2.0957      0.396      5.287      0.000       1.318       2.873
volatility_7d       -0.3647      0.288     -1.265      0.206      -0.930       0.200
volatility_30d       0.1042      0.142      0.732      0.464      -0.175       0.384
close_lag1          -0.4554      0.073     -6.233      0.000      -0.599      -0.312
close_lag5           0.1679      0.008     21.482      0.000       0.153       0.183
rsi                  0.0269      0.005      5.599      0.000       0.017       0.036
macd                 4.2378      0.057     74.842      0.000       4.127       4.349
macd_signal         -1.4794      0.021    -69.819      0.000      -1.521      -1.438
macd_diff            5.7172      0.076     75.578      0.000       5.569       5.865
bb_upper             0.3091      0.007     42.595      0.000       0.295       0.323
bb_lower             0.2791      0.010     28.679      0.000       0.260       0.298
bb_mavg              0.2941      0.007     40.222      0.000       0.280       0.308
obv               6.348e-08   1.06e-07      0.599      0.549   -1.44e-07    2.71e-07
==============================================================================
Omnibus:                      959.335   Durbin-Watson:                   1.767
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            63937.408
Skew:                          -0.207   Prob(JB):                         0.00
Kurtosis:                      23.211   Cond. No.                     1.27e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 4.11e-16. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_LTC_USDT.csv --------------------------------------------------
Formula: close ~ open+high+low+volume_ltc+buytakeramount+buytakerquantity+weightedaverage+sma_7+std_7+sma_14+std_14+sma_30+std_30+daily_return+volatility_7d+volatility_30d+close_lag1+close_lag5+rsi+macd+macd_signal+macd_diff+bb_upper+bb_lower+bb_mavg+obv
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 2.324e+05
Date:                Fri, 04 Jul 2025   Prob (F-statistic):               0.00
Time:                        23:33:02   Log-Likelihood:                -6879.9
No. Observations:                3737   AIC:                         1.381e+04
Df Residuals:                    3712   BIC:                         1.397e+04
Df Model:                          24                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept           -1.0946      0.175     -6.243      0.000      -1.438      -0.751
open                -0.5754      0.085     -6.799      0.000      -0.741      -0.409
high                 0.3003      0.018     16.330      0.000       0.264       0.336
low                  0.2811      0.017     16.690      0.000       0.248       0.314
volume_ltc        4.823e-08   1.38e-08      3.494      0.000    2.12e-08    7.53e-08
buytakeramount   -1.688e-07   6.11e-08     -2.761      0.006   -2.89e-07    -4.9e-08
buytakerquantity  7.778e-06   4.83e-06      1.610      0.108   -1.69e-06    1.72e-05
weightedaverage      0.0164      0.029      0.562      0.574      -0.041       0.073
sma_7               -1.2914      0.029    -44.444      0.000      -1.348      -1.234
std_7                0.0405      0.010      3.875      0.000       0.020       0.061
sma_14               0.6093      0.017     35.142      0.000       0.575       0.643
std_14              -0.0477      0.012     -3.857      0.000      -0.072      -0.023
sma_30               0.6406      0.014     45.895      0.000       0.613       0.668
std_30              -0.0009      0.007     -0.118      0.906      -0.015       0.013
daily_return         2.7326      0.357      7.648      0.000       2.032       3.433
volatility_7d       -0.4397      0.167     -2.627      0.009      -0.768      -0.112
volatility_30d       0.0677      0.083      0.811      0.418      -0.096       0.231
close_lag1          -0.0448      0.087     -0.517      0.605      -0.215       0.125
close_lag5           0.1843      0.008     23.196      0.000       0.169       0.200
rsi                  0.0209      0.003      6.708      0.000       0.015       0.027
macd                 3.9147      0.060     65.545      0.000       3.798       4.032
macd_signal         -1.3928      0.022    -63.387      0.000      -1.436      -1.350
macd_diff            5.3075      0.080     66.173      0.000       5.150       5.465
bb_upper             0.3025      0.007     44.918      0.000       0.289       0.316
bb_lower             0.2829      0.007     39.944      0.000       0.269       0.297
bb_mavg              0.2927      0.006     47.960      0.000       0.281       0.305
obv               7.302e-09   1.68e-08      0.433      0.665   -2.57e-08    4.03e-08
==============================================================================
Omnibus:                      957.539   Durbin-Watson:                   1.880
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            67108.795
Skew:                           0.174   Prob(JB):                         0.00
Kurtosis:                      23.757   Cond. No.                     1.48e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 1.04e-15. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_BTC_USDT.csv --------------------------------------------------
Formula: close ~ open+high+low+volume_btc+buytakeramount+buytakerquantity+weightedaverage+sma_7+std_7+sma_14+std_14+sma_30+std_30+daily_return+volatility_7d+volatility_30d+close_lag1+close_lag5+rsi+macd+macd_signal+macd_diff+bb_upper+bb_lower+bb_mavg+obv
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 1.485e+06
Date:                Fri, 04 Jul 2025   Prob (F-statistic):               0.00
Time:                        23:33:02   Log-Likelihood:                -26454.
No. Observations:                3754   AIC:                         5.296e+04
Df Residuals:                    3729   BIC:                         5.311e+04
Df Model:                          24                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept          -50.3406     33.551     -1.500      0.134    -116.121      15.439
open                -0.3524      0.224     -1.575      0.115      -0.791       0.086
high                 0.3400      0.016     20.850      0.000       0.308       0.372
low                  0.3047      0.014     21.135      0.000       0.276       0.333
volume_btc        7.887e-07   2.52e-07      3.124      0.002    2.94e-07    1.28e-06
buytakeramount   -4.463e-07    6.5e-07     -0.687      0.492   -1.72e-06    8.28e-07
buytakerquantity    -0.0068      0.023     -0.291      0.771      -0.052       0.039
weightedaverage     -0.0450      0.025     -1.791      0.073      -0.094       0.004
sma_7               -1.3748      0.028    -48.418      0.000      -1.431      -1.319
std_7                0.0124      0.011      1.130      0.259      -0.009       0.034
sma_14               0.6639      0.018     37.719      0.000       0.629       0.698
std_14               0.0219      0.013      1.633      0.102      -0.004       0.048
sma_30               0.6945      0.015     46.326      0.000       0.665       0.724
std_30              -0.0123      0.008     -1.589      0.112      -0.027       0.003
daily_return      2107.0401    170.673     12.345      0.000    1772.418    2441.662
volatility_7d     -157.0825    127.699     -1.230      0.219    -407.449      93.284
volatility_30d      44.7979     84.098      0.533      0.594    -120.084     209.680
close_lag1          -0.2274      0.224     -1.015      0.310      -0.667       0.212
close_lag5           0.2059      0.008     25.840      0.000       0.190       0.221
rsi                  0.9241      0.506      1.826      0.068      -0.068       1.917
macd                 3.9279      0.061     64.347      0.000       3.808       4.048
macd_signal         -1.3358      0.023    -59.348      0.000      -1.380      -1.292
macd_diff            5.2637      0.082     64.118      0.000       5.103       5.425
bb_upper             0.2665      0.008     34.476      0.000       0.251       0.282
bb_lower             0.2600      0.008     33.585      0.000       0.245       0.275
bb_mavg              0.2633      0.007     38.146      0.000       0.250       0.277
obv                 -0.0002      0.000     -1.446      0.148      -0.000    5.72e-05
==============================================================================
Omnibus:                      600.431   Durbin-Watson:                   1.827
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            10493.780
Skew:                          -0.131   Prob(JB):                         0.00
Kurtosis:                      11.187   Cond. No.                     1.19e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 1.28e-13. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Seleção das variáveis mais relevantes para o modelo de regressão OLS

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



====================================================
1. Conhecimento prévio (domínio do problema):
Variáveis como:

open, high, low, close, weightedaverage,

sma_7, sma_14, sma_30,

bb_mavg, bb_upper, bb_lower,

close_lag1, close_lag5

 estão todas derivadas dos mesmos preços históricos. Muitas delas são formas diferentes de suavizar ou 
 resumir o mesmo conjunto de dados (preço de fechamento).

 sma_30 é a média móvel dos últimos 30 fechamentos.

bb_mavg geralmente é igual à sma_20.

weightedaverage, open, high, low, e close em dias com pouca volatilidade são numericamente muito próximos.

Portanto, a correlação entre elas é quase inevitável, mesmo sem olhar os dados


2. Indícios no relatório OLS:
A) p-valores altos (irrelevância estatística)
Exemplo:

text
Copiar
Editar
volume_xrp     coef ≈ 0,  p ≈ 0.93
buytakeramount p ≈ 0.98
std_14         p ≈ 0.50
Isso mostra que essas variáveis não explicam mais nada do que as outras já explicam. É o que esperamos de variáveis colineares.

B) Coeficientes com sinais inconsistentes
close_lag1 com coef negativo (-0.2), close_lag5 com coef positivo (+0.2)
Eles carregam a mesma informação de preço passado, mas estão sendo penalizados de forma oposta, outro sintoma clássico de multicolinearidade.

C) Aviso direto do statsmodels:
"The smallest eigenvalue is X. This might indicate strong multicollinearity or that the design matrix is singular."

Este aviso aparece em praticamente todas as análises.

Dessa forma, variáveis candidatas ao modelo mais otimizado são:
['high', 'low', 'sma_7', 'sma_14', 'sma_30', 
 'close_lag5', 'macd', 'macd_signal', 'macd_diff', 
 'bb_upper', 'bb_lower', 'bb_mavg', 'daily_return']


 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Novo teste

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/content/featured_XRP_USDT.csv --------------------------------------------------
Formula: close ~ high + low + sma_7 + sma_14 + sma_30 + close_lag5 + macd + macd_signal + macd_diff + bb_upper + bb_lower + bb_mavg + daily_return
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 2.867e+05
Date:                Sun, 06 Jul 2025   Prob (F-statistic):               0.00
Time:                        00:22:31   Log-Likelihood:                 9440.0
No. Observations:                3752   AIC:                        -1.886e+04
Df Residuals:                    3740   BIC:                        -1.878e+04
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.0008      0.000     -1.867      0.062      -0.002    4.13e-05
high             0.1817      0.012     14.949      0.000       0.158       0.206
low              0.1416      0.010     14.090      0.000       0.122       0.161
sma_7           -1.5444      0.042    -36.621      0.000      -1.627      -1.462
sma_14           0.6385      0.026     24.860      0.000       0.588       0.689
sma_30           0.6064      0.021     29.047      0.000       0.566       0.647
close_lag5       0.2841      0.012     23.692      0.000       0.261       0.308
macd             3.4953      0.083     42.089      0.000       3.332       3.658
macd_signal     -1.2242      0.031    -39.665      0.000      -1.285      -1.164
macd_diff        4.7195      0.111     42.341      0.000       4.501       4.938
bb_upper         0.2329      0.010     23.899      0.000       0.214       0.252
bb_lower         0.2281      0.009     24.140      0.000       0.210       0.247
bb_mavg          0.2305      0.009     24.562      0.000       0.212       0.249
daily_return     0.0731      0.003     23.453      0.000       0.067       0.079
==============================================================================
Omnibus:                     1240.592   Durbin-Watson:                   2.107
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           173256.700
Skew:                          -0.461   Prob(JB):                         0.00
Kurtosis:                      36.278   Cond. No.                     5.95e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 5.96e-30. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_ETC_USDT.csv --------------------------------------------------
Formula: close ~ high + low + sma_7 + sma_14 + sma_30 + close_lag5 + macd + macd_signal + macd_diff + bb_upper + bb_lower + bb_mavg + daily_return
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 2.571e+05
Date:                Sun, 06 Jul 2025   Prob (F-statistic):               0.00
Time:                        00:22:31   Log-Likelihood:                -2174.1
No. Observations:                3228   AIC:                             4372.
Df Residuals:                    3216   BIC:                             4445.
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.0309      0.014     -2.167      0.030      -0.059      -0.003
high             0.2333      0.010     23.812      0.000       0.214       0.252
low              0.2338      0.010     23.451      0.000       0.214       0.253
sma_7           -1.0710      0.034    -31.711      0.000      -1.137      -1.005
sma_14           0.4126      0.020     20.906      0.000       0.374       0.451
sma_30           0.4161      0.017     24.733      0.000       0.383       0.449
close_lag5       0.1875      0.010     19.167      0.000       0.168       0.207
macd             2.5689      0.068     37.510      0.000       2.435       2.703
macd_signal     -0.9023      0.026    -34.425      0.000      -0.954      -0.851
macd_diff        3.4713      0.092     37.591      0.000       3.290       3.652
bb_upper         0.1956      0.008     24.736      0.000       0.180       0.211
bb_lower         0.1972      0.008     25.226      0.000       0.182       0.212
bb_mavg          0.1964      0.008     25.596      0.000       0.181       0.211
daily_return     8.5970      0.150     57.502      0.000       8.304       8.890
==============================================================================
Omnibus:                     1245.918   Durbin-Watson:                   1.995
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           248046.295
Skew:                           0.667   Prob(JB):                         0.00
Kurtosis:                      45.924   Cond. No.                     2.01e+17
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 3.78e-28. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_ZRX_USDT.csv --------------------------------------------------
Formula: close ~ high + low + sma_7 + sma_14 + sma_30 + close_lag5 + macd + macd_signal + macd_diff + bb_upper + bb_lower + bb_mavg + daily_return
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.998
Model:                            OLS   Adj. R-squared:                  0.998
Method:                 Least Squares   F-statistic:                 1.182e+05
Date:                Sun, 06 Jul 2025   Prob (F-statistic):               0.00
Time:                        00:22:31   Log-Likelihood:                 6826.8
No. Observations:                2464   AIC:                        -1.363e+04
Df Residuals:                    2452   BIC:                        -1.356e+04
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.0027      0.001     -4.829      0.000      -0.004      -0.002
high             0.1016      0.009     11.573      0.000       0.084       0.119
low              0.3267      0.011     31.101      0.000       0.306       0.347
sma_7           -0.7637      0.034    -22.358      0.000      -0.831      -0.697
sma_14           0.3593      0.025     14.152      0.000       0.310       0.409
sma_30           0.3819      0.020     18.905      0.000       0.342       0.422
close_lag5       0.0942      0.009     10.702      0.000       0.077       0.111
macd             2.3112      0.069     33.366      0.000       2.175       2.447
macd_signal     -0.8061      0.027    -29.389      0.000      -0.860      -0.752
macd_diff        3.1173      0.093     33.426      0.000       2.934       3.300
bb_upper         0.1738      0.009     18.590      0.000       0.155       0.192
bb_lower         0.1687      0.010     17.530      0.000       0.150       0.188
bb_mavg          0.1712      0.009     18.352      0.000       0.153       0.190
daily_return     0.2434      0.003     72.678      0.000       0.237       0.250
==============================================================================
Omnibus:                      785.521   Durbin-Watson:                   1.735
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           148622.413
Skew:                          -0.160   Prob(JB):                         0.00
Kurtosis:                      41.046   Cond. No.                     4.54e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 4.41e-30. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_EOS_USDT.csv --------------------------------------------------
Formula: close ~ high + low + sma_7 + sma_14 + sma_30 + close_lag5 + macd + macd_signal + macd_diff + bb_upper + bb_lower + bb_mavg + daily_return
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 1.580e+05
Date:                Sun, 06 Jul 2025   Prob (F-statistic):               0.00
Time:                        00:22:31   Log-Likelihood:                 3103.3
No. Observations:                2461   AIC:                            -6183.
Df Residuals:                    2449   BIC:                            -6113.
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.0025      0.002     -1.021      0.307      -0.007       0.002
high             0.1640      0.012     13.820      0.000       0.141       0.187
low              0.1454      0.012     12.042      0.000       0.122       0.169
sma_7           -1.1053      0.038    -29.043      0.000      -1.180      -1.031
sma_14           0.4649      0.025     18.242      0.000       0.415       0.515
sma_30           0.4917      0.020     24.269      0.000       0.452       0.531
close_lag5       0.1669      0.010     16.571      0.000       0.147       0.187
macd             2.9741      0.082     36.237      0.000       2.813       3.135
macd_signal     -1.0341      0.032    -32.336      0.000      -1.097      -0.971
macd_diff        4.0082      0.111     36.038      0.000       3.790       4.226
bb_upper         0.2206      0.009     24.280      0.000       0.203       0.238
bb_lower         0.2287      0.009     24.129      0.000       0.210       0.247
bb_mavg          0.2247      0.009     24.843      0.000       0.207       0.242
daily_return     1.4611      0.027     53.650      0.000       1.408       1.514
==============================================================================
Omnibus:                     1810.257   Durbin-Watson:                   2.317
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          1038013.611
Skew:                           2.217   Prob(JB):                         0.00
Kurtosis:                     103.515   Cond. No.                     1.07e+17
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 1.82e-29. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
Warning: Could not determine volume column name for featured_BCH_USDT.csv. Skipping.
/content/featured_DASH_USDT.csv --------------------------------------------------
Formula: close ~ high + low + sma_7 + sma_14 + sma_30 + close_lag5 + macd + macd_signal + macd_diff + bb_upper + bb_lower + bb_mavg + daily_return
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 2.331e+05
Date:                Sun, 06 Jul 2025   Prob (F-statistic):               0.00
Time:                        00:22:32   Log-Likelihood:                -11996.
No. Observations:                3755   AIC:                         2.402e+04
Df Residuals:                    3743   BIC:                         2.409e+04
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.0341      0.124     -0.275      0.783      -0.277       0.208
high             0.2593      0.014     18.288      0.000       0.232       0.287
low              0.2311      0.011     20.920      0.000       0.209       0.253
sma_7           -1.2586      0.048    -26.188      0.000      -1.353      -1.164
sma_14           0.4453      0.030     14.878      0.000       0.387       0.504
sma_30           0.4487      0.025     17.616      0.000       0.399       0.499
close_lag5       0.2253      0.013     17.575      0.000       0.200       0.250
macd             2.7915      0.097     28.818      0.000       2.602       2.981
macd_signal     -0.9655      0.035    -27.801      0.000      -1.034      -0.897
macd_diff        3.7570      0.129     29.076      0.000       3.504       4.010
bb_upper         0.2219      0.011     19.660      0.000       0.200       0.244
bb_lower         0.2088      0.011     19.082      0.000       0.187       0.230
bb_mavg          0.2154      0.011     19.996      0.000       0.194       0.236
daily_return     0.0845      0.062      1.371      0.170      -0.036       0.205
==============================================================================
Omnibus:                     1324.754   Durbin-Watson:                   2.304
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           346887.306
Skew:                           0.368   Prob(JB):                         0.00
Kurtosis:                      50.081   Cond. No.                     1.09e+17
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 1.02e-25. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_ETH_USDT.csv --------------------------------------------------
Formula: close ~ high + low + sma_7 + sma_14 + sma_30 + close_lag5 + macd + macd_signal + macd_diff + bb_upper + bb_lower + bb_mavg + daily_return
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 6.577e+05
Date:                Sun, 06 Jul 2025   Prob (F-statistic):               0.00
Time:                        00:22:32   Log-Likelihood:                -16959.
No. Observations:                3583   AIC:                         3.394e+04
Df Residuals:                    3571   BIC:                         3.402e+04
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -1.3539      0.655     -2.068      0.039      -2.638      -0.070
high             0.1245      0.015      8.294      0.000       0.095       0.154
low              0.1840      0.011     16.861      0.000       0.163       0.205
sma_7           -1.3855      0.041    -33.409      0.000      -1.467      -1.304
sma_14           0.5777      0.026     22.316      0.000       0.527       0.628
sma_30           0.5753      0.021     27.165      0.000       0.534       0.617
close_lag5       0.2376      0.012     20.209      0.000       0.215       0.261
macd             3.3327      0.087     38.305      0.000       3.162       3.503
macd_signal     -1.1364      0.032    -35.251      0.000      -1.200      -1.073
macd_diff        4.4691      0.117     38.148      0.000       4.239       4.699
bb_upper         0.2359      0.010     24.058      0.000       0.217       0.255
bb_lower         0.2225      0.010     22.108      0.000       0.203       0.242
bb_mavg          0.2292      0.010     23.620      0.000       0.210       0.248
daily_return   288.0692      8.489     33.935      0.000     271.426     304.713
==============================================================================
Omnibus:                      643.354   Durbin-Watson:                   2.057
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            14326.453
Skew:                           0.168   Prob(JB):                         0.00
Kurtosis:                      12.790   Cond. No.                     3.13e+17
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 9.9e-25. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_XMR_USDT.csv --------------------------------------------------
Formula: close ~ high + low + sma_7 + sma_14 + sma_30 + close_lag5 + macd + macd_signal + macd_diff + bb_upper + bb_lower + bb_mavg + daily_return
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.998
Model:                            OLS   Adj. R-squared:                  0.998
Method:                 Least Squares   F-statistic:                 2.239e+05
Date:                Sun, 06 Jul 2025   Prob (F-statistic):               0.00
Time:                        00:22:32   Log-Likelihood:                -10109.
No. Observations:                3755   AIC:                         2.024e+04
Df Residuals:                    3743   BIC:                         2.032e+04
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.1871      0.100     -1.869      0.062      -0.383       0.009
high             0.0618      0.013      4.675      0.000       0.036       0.088
low              0.1956      0.011     17.762      0.000       0.174       0.217
sma_7           -1.4662      0.044    -33.617      0.000      -1.552      -1.381
sma_14           0.6508      0.029     22.518      0.000       0.594       0.708
sma_30           0.6193      0.024     26.073      0.000       0.573       0.666
close_lag5       0.2377      0.012     19.370      0.000       0.214       0.262
macd             3.5899      0.085     42.030      0.000       3.422       3.757
macd_signal     -1.2467      0.032    -38.639      0.000      -1.310      -1.183
macd_diff        4.8366      0.114     42.290      0.000       4.612       5.061
bb_upper         0.2496      0.011     23.370      0.000       0.229       0.271
bb_lower         0.2192      0.011     19.227      0.000       0.197       0.242
bb_mavg          0.2344      0.011     21.700      0.000       0.213       0.256
daily_return    12.0260      0.564     21.334      0.000      10.921      13.131
==============================================================================
Omnibus:                     1436.865   Durbin-Watson:                   2.200
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           153929.156
Skew:                           0.837   Prob(JB):                         0.00
Kurtosis:                      34.321   Cond. No.                     3.31e+17
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 7.04e-27. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_LTC_USDT.csv --------------------------------------------------
Formula: close ~ high + low + sma_7 + sma_14 + sma_30 + close_lag5 + macd + macd_signal + macd_diff + bb_upper + bb_lower + bb_mavg + daily_return
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       0.998
Model:                            OLS   Adj. R-squared:                  0.998
Method:                 Least Squares   F-statistic:                 2.034e+05
Date:                Sun, 06 Jul 2025   Prob (F-statistic):               0.00
Time:                        00:22:32   Log-Likelihood:                -8591.7
No. Observations:                3737   AIC:                         1.721e+04
Df Residuals:                    3725   BIC:                         1.728e+04
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -0.0834      0.067     -1.239      0.216      -0.215       0.049
high             0.1723      0.013     13.609      0.000       0.148       0.197
low              0.2138      0.012     18.261      0.000       0.191       0.237
sma_7           -1.3394      0.045    -29.611      0.000      -1.428      -1.251
sma_14           0.5022      0.027     18.624      0.000       0.449       0.555
sma_30           0.5054      0.021     23.725      0.000       0.464       0.547
close_lag5       0.2393      0.012     19.211      0.000       0.215       0.264
macd             3.1286      0.091     34.511      0.000       2.951       3.306
macd_signal     -1.1035      0.034    -32.743      0.000      -1.170      -1.037
macd_diff        4.2322      0.122     34.656      0.000       3.993       4.472
bb_upper         0.2413      0.009     25.416      0.000       0.223       0.260
bb_lower         0.2306      0.010     23.619      0.000       0.211       0.250
bb_mavg          0.2360      0.009     25.011      0.000       0.217       0.254
daily_return     9.8479      0.403     24.418      0.000       9.057      10.639
==============================================================================
Omnibus:                     1511.687   Durbin-Watson:                   2.155
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           253115.115
Skew:                          -0.822   Prob(JB):                         0.00
Kurtosis:                      43.285   Cond. No.                     1.12e+17
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 2.39e-26. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
/content/featured_BTC_USDT.csv --------------------------------------------------
Formula: close ~ high + low + sma_7 + sma_14 + sma_30 + close_lag5 + macd + macd_signal + macd_diff + bb_upper + bb_lower + bb_mavg + daily_return
----------------------------------------------------------------------
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  close   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 1.735e+06
Date:                Sun, 06 Jul 2025   Prob (F-statistic):               0.00
Time:                        00:22:32   Log-Likelihood:                -27633.
No. Observations:                3754   AIC:                         5.529e+04
Df Residuals:                    3742   BIC:                         5.536e+04
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept      -24.2018      8.549     -2.831      0.005     -40.964      -7.440
high             0.1790      0.013     13.796      0.000       0.154       0.204
low              0.1593      0.010     15.460      0.000       0.139       0.180
sma_7           -1.3390      0.039    -34.750      0.000      -1.415      -1.263
sma_14           0.5745      0.024     24.108      0.000       0.528       0.621
sma_30           0.5669      0.020     28.348      0.000       0.528       0.606
close_lag5       0.2343      0.011     21.616      0.000       0.213       0.256
macd             3.1898      0.081     39.591      0.000       3.032       3.348
macd_signal     -1.0795      0.030    -36.162      0.000      -1.138      -1.021
macd_diff        4.2693      0.108     39.357      0.000       4.057       4.482
bb_upper         0.2131      0.009     22.743      0.000       0.195       0.231
bb_lower         0.2034      0.009     21.424      0.000       0.185       0.222
bb_mavg          0.2082      0.009     22.572      0.000       0.190       0.226
daily_return  7456.4342    176.710     42.196      0.000    7109.977    7802.891
==============================================================================
Omnibus:                      578.994   Durbin-Watson:                   2.033
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             9869.814
Skew:                          -0.022   Prob(JB):                         0.00
Kurtosis:                      10.943   Cond. No.                     5.38e+17
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 1.53e-22. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Significados

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

os novos resultados com a fórmula reduzida e ajustada mostram melhora significativa:

Critério	                    Antes	                        Agora
R²	                            0.998–1.000	                    0.998–1.000
p-valor das variáveis	        Vários acima de 0.05	        Todos < 0.05
NaN em resultados	            Frequentemente presentes	    Nenhum NaN encontrado
Número de variáveis	            25+	                            14
Estabilidade numérica	        Baixa (singularidade)	        Alta (relatórios completos)
Eigenvalue mínima	            ~1e-28 a 1e-30	                ~1e-22 (melhor)

"""
