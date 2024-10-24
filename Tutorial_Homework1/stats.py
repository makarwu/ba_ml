"@makarwu"

from scipy import stats
import numpy as np

# T1.2
"""before = [27, 31, 23, 35, 26, 27, 26, 18, 22, 21] 
after = [40, 36, 43, 34, 25, 41, 32, 29, 21, 36]

t_statistic, p_value = stats.ttest_rel(before, after)

print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")"""

# T1.3

adults = [4, 2, 3, 5, 7, 2, 7, 3, 5, 2]
children = [2, 1, 5, 3, 1, 3, 2, 3]

mean_adults = np.mean(adults)
mean_children = np.mean(children)

t_statistic, p_value = stats.ttest_ind(adults, children, alternative='greater')

print(f"Adult mean: {mean_adults}")
print(f"Children mean: {mean_children}")
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")