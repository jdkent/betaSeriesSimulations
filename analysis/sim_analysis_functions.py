"""simulation analysis functions"""
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind, ttest_1samp
import numpy as np
import pandas as pd


def add_targets(g, bound=0.01, width=0.5, targets=None):
    """adds rectangles to each ground truth correlation"""
    if targets is None:
        targets = [float(x.get_text()) for x in g.get_xticklabels()]
    for idx, target in enumerate(targets):
        trans = g.transData
        g.add_patch(Rectangle((idx - (width/2), target - bound),
                              width, bound*2, color='red',
                              zorder=100, transform=trans))

    return g


def test_power(df, estimation_method="lss", iti_mean=4.0,
               n_trials=30, correlation_tgt1=0.0, correlation_tgt2=0.3,
               signal_magnitude=1, simulations=10000,
               trial_type1=None, trial_type2=None,
               sample_size=40, trial_var=1):
    """run a power analysis on the simulated data"""

    base_query = ('(estimation_method == "{em}") &'
                  '(iti_mean == {iti}) &'
                  '(n_trials == {trl}) &'
                  '(cnr == {snr}) &'
                  '(avnr == {tv})').format(
                    em=estimation_method,
                    iti=iti_mean,
                    trl=n_trials,
                    snr=signal_magnitude,
                    tv=trial_var,
                  )
    group1_query = base_query + \
        '& (correlation_target == {tgt})'.format(
            tgt=correlation_tgt1)
    if trial_type1:
        group1_query = group1_query + \
            '& (trial_type == "{tp}")'.format(tp=trial_type1)

    group2_query = base_query + \
        '& (correlation_target == {tgt})'.format(
            tgt=correlation_tgt2)

    if trial_type2:
        group2_query = group2_query + \
            '& (trial_type == "{tp}")'.format(tp=trial_type2)

    group1 = df.query(group1_query)
    group2 = df.query(group2_query)
    target_diff = abs(correlation_tgt2 - correlation_tgt1)
    test_collector = {"t_value": [], "p_value": [], "estimate": [],
                      "tgt_corr_diff": [target_diff] * simulations,
                      "avnr": [trial_var] * simulations,
                      "estimation_method": [estimation_method] * simulations,
                      "iti_mean": [iti_mean] * simulations,
                      "n_trials": [n_trials] * simulations,
                      "cnr": [signal_magnitude] * simulations}

    for _ in range(simulations):
        if correlation_tgt1 == correlation_tgt2:
            overall_sample = np.random.choice(
                group1['corr_obs_trans'].values,
                int(sample_size*2), replace=False)
            group1_sample = overall_sample[0:sample_size]
            group2_sample = overall_sample[sample_size:]
        else:
            group1_sample = np.random.choice(group1['corr_obs_trans'].values, sample_size, replace=False)
            group2_sample = np.random.choice(group2['corr_obs_trans'].values, sample_size, replace=False)
        sample = group1_sample - group2_sample
        test_collector['estimate'].append(np.abs(sample.mean()))
        t, p = ttest_1samp(sample, 0)
        test_collector["t_value"].append(t)
        if correlation_tgt1 < correlation_tgt2 and t > 0 and p < 0.05:
            test_collector["p_value"].append(1-p)
        elif correlation_tgt1 > correlation_tgt2 and t < 0 and p < 0.05:
            test_collector["p_value"].append(1-p)
        else:
            test_collector["p_value"].append(p)

    test_df = pd.DataFrame.from_dict(test_collector)
    pwr = np.sum(test_df["p_value"] < 0.05) / simulations

    return test_df, pwr


# https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height() * 100)
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

    return axs


def test_power2(df, estimation_method="lss", iti_mean=4.0,
                n_trials=30, corr_diff=0.1,
                signal_magnitude=1, simulations=10000,
                trial_type1='c0', trial_type2='c1',
                sample_size=40, trial_var=1):
    """run a power analysis on the simulated data"""

    base_query = ('(estimation_method == "{em}") &'
                  '(iti_mean == {iti}) &'
                  '(n_trials == {trl}) &'
                  '(cnr == {snr}) &'
                  '(avnr == {tv})').format(
                    em=estimation_method,
                    iti=iti_mean,
                    trl=n_trials,
                    snr=signal_magnitude,
                    tv=trial_var,
                  )
    group1_query = base_query + \
        '& (correlation_target == {tgt})'.format(
            tgt=correlation_tgt1)
    if trial_type1:
        group1_query = group1_query + \
            '& (trial_type == "{tp}")'.format(tp=trial_type1)

    group2_query = base_query + \
        '& (correlation_target == {tgt})'.format(
            tgt=correlation_tgt2)

    if trial_type2:
        group2_query = group2_query + \
            '& (trial_type == "{tp}")'.format(tp=trial_type2)

    group1 = df.query(group1_query)
    group2 = df.query(group2_query)
    target_diff = abs(correlation_tgt2 - correlation_tgt1)
    test_collector = {"t_value": [], "p_value": [], "estimate": [],
                      "tgt_corr_diff": [target_diff] * simulations,
                      "avnr": [trial_var] * simulations,
                      "estimation_method": [estimation_method] * simulations,
                      "iti_mean": [iti_mean] * simulations,
                      "n_trials": [n_trials] * simulations,
                      "cnr": [signal_magnitude] * simulations}

    for _ in range(simulations):
        if correlation_tgt1 == correlation_tgt2:
            overall_sample = np.random.choice(
                group1['corr_obs_trans'].values,
                int(sample_size*2), replace=False)
            group1_sample = overall_sample[0:sample_size]
            group2_sample = overall_sample[sample_size:]
        else:
            group1_sample = np.random.choice(group1['corr_obs_trans'].values, sample_size, replace=False)
            group2_sample = np.random.choice(group2['corr_obs_trans'].values, sample_size, replace=False)
        sample = group1_sample - group2_sample
        test_collector['estimate'].append(np.abs(sample.mean()))
        t, p = ttest_1samp(sample, 0)
        test_collector["t_value"].append(t)
        if correlation_tgt1 < correlation_tgt2 and t > 0 and p < 0.05:
            test_collector["p_value"].append(1-p)
        elif correlation_tgt1 > correlation_tgt2 and t < 0 and p < 0.05:
            test_collector["p_value"].append(1-p)
        else:
            test_collector["p_value"].append(p)

    test_df = pd.DataFrame.from_dict(test_collector)
    pwr = np.sum(test_df["p_value"] < 0.05) / simulations

    return test_df, pwr