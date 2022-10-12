import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

scaling = 12 * 12 * 2

df = pd.read_csv(r'/Users/christianpavilanis/Desktop/School/UChicago/MSCI_sample.csv')
df_rf = pd.read_csv(r'/Users/christianpavilanis/Desktop/School/UChicago/F-F_Research_Data_Factors.csv')
df_rf.set_index('date', inplace=True)

## adding market cap and greenness scores
df["MKTCAP"] = df["PRC"] * df["SHROUT"]
df['G'] = -(10 - df['ENVIRONMENTAL_PILLAR_SCORE']) * df['ENVIRONMENTAL_PILLAR_WEIGHT'] / 100

## calculating Return of each stock. obviously get NaN for first one
df_sorted = df.sort_values(['ISSUERID', 'AS_OF_DATE'])
df_sorted['RETURN'] = df_sorted.groupby('ISSUERID')['PRC'].pct_change()

## re-sort chronologicall, remove 2012, and remove after 2021-01
df_sorted = df_sorted.sort_values('AS_OF_DATE').tail(len(df_sorted) - 1807)
df_sorted = df_sorted.sort_values('AS_OF_DATE').head(len(df_sorted) - 5669)

## found the percentile for greennest stocks and the brownest stocks
df_sorted['AS_OF_DATE'] = pd.to_datetime(df_sorted['AS_OF_DATE'])

## note: the quantile function in python here is not returning equal lengths for the upper and lower third, so I'm
## forcing the dimension to agree with the extra term on line 29
df_sorted['G_low'] = (df_sorted.groupby(df_sorted['AS_OF_DATE'].
                                        dt.to_period('M'))['G'].
                      transform(lambda x: x.quantile(1 / 3 - 0.002565)))
df_sorted['G_high'] = (df_sorted.groupby(df_sorted['AS_OF_DATE'].
                                         dt.to_period('M'))['G'].
                       transform(lambda x: x.quantile(2 / 3)))

## found the total market cap for each group (this is used for value weighting)
df_sorted['total_cap'] = (df_sorted.groupby(df_sorted['AS_OF_DATE'].
                                            dt.to_period('M'))['MKTCAP'].transform('sum'))

## get the green stocks
df_sorted['delta'] = df_sorted['G'] - df_sorted['G_high']
green = df_sorted[df_sorted['delta'] > 0]

## get the brown stocks
df_sorted['delta'] = df_sorted['G_low'] - df_sorted['G']
brown = df_sorted[df_sorted['delta'] >= 0]

## calculated value weighting
green['value_weight'] = green['MKTCAP'] / green['total_cap']
green['return_weighted'] = green['RETURN'] * green['value_weight']
brown['value_weight'] = brown['MKTCAP'] / brown['total_cap']
brown['return_weighted'] = brown['RETURN'] * brown['value_weight']

green['Green'] = ((green['return_weighted'] + 1).cumprod() - 1)
brown['Brown'] = ((brown['return_weighted'] + 1).cumprod() - 1)

green['date'] = green['AS_OF_DATE']
brown['date'] = brown['AS_OF_DATE']

y = green['Green'] * scaling
z = brown['Brown'] * scaling
X = green['date']

plt.plot(X, y, color='#009900', label='Green')
plt.plot(X, z, color='#cc6600', label='Brown', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Cumulative return (%)")

plt.legend(loc='upper left', frameon=False)

plt.show()

print(green['Green'])
print(df_rf.head(5))

# green_group = green.groupby(green['AS_OF_DATE'])['return_weighted']+1

green['return_weighted'] = green['return_weighted'] + 1
green['CUM'] = green.groupby('AS_OF_DATE')['return_weighted'].cumprod()
brown['return_weighted'] = brown['return_weighted'] + 1
brown['CUM'] = brown.groupby('AS_OF_DATE')['return_weighted'].cumprod()

print('next.....')

green['cum_max'] = green.groupby('date')['CUM'].transform('max')
green_monthly_returns = green.groupby('date')['CUM'].last()
green_monthly_returns = green_monthly_returns.to_frame()

green_array = green_monthly_returns[['CUM']].values

brown['cum_max'] = brown.groupby('date')['CUM'].transform('max')
brown_monthly_returns = brown.groupby('date')['CUM'].last()
brown_monthly_returns = brown_monthly_returns.to_frame()

brown_array = brown_monthly_returns[['CUM']].values

rf_array = df_rf[['RF']].values

green_excess = [(green_array - 1) * 100 - rf_array]
brown_excess = [(brown_array - 1) * 100 - rf_array]
green_minus_brown = np.subtract(green_excess, brown_excess)

Sharpe_green_minus_brown = np.mean(green_minus_brown) / (np.std(green_minus_brown))
Sharpe_green = np.mean(green_excess) / (np.std(green_excess))
Sharpe_brown = np.mean(brown_excess) / (np.std(brown_excess))
# print(Sharpe_green)
# print(Sharpe_brown)
print(np.mean(green_minus_brown))
print(Sharpe_green_minus_brown)
# print(np.std(green_excess))


# green_monthly_returns.to_excel('/Users/christianpavilanis/Desktop/School/UChicago/green.xlsx')
df_rf.to_excel('/Users/christianpavilanis/Desktop/School/UChicago/risk_free.xlsx')
