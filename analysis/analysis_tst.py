
#%%
import os
from subprocess import call
from textwrap import dedent

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_1samp

import sim_analysis_functions as saf
sns.set(style="ticks", rc={"lines.linewidth": 3.0})
sns.set(font_scale=1.3)
sns.set_style({'axes.grid': False})
# plt.style.use("analysis/gray_background.mplstyle")
def save_eps(fig, fname):
    fig.savefig(fname + '.pdf', bbox_inches='tight')
    call(["pdf2ps", fname + '.pdf', fname + '.eps'])
    os.remove(fname + '.pdf')
#%%
import importlib
importlib.reload(saf)

#%%
df = pd.read_csv('../simulation_200_real_params.tsv', sep='\t')
df.loc[df['correlation_observed'] == 1, 'correlation_observed'] -= 0.0001
df.loc[df['correlation_observed'] == -1, 'correlation_observed'] += 0.0001
df['corr_obs_trans'] = np.arctanh(df['correlation_observed'])
df['corr_obs_trans_clip'] = np.clip(df['corr_obs_trans'], -1.5, 1.5)
# Some simulations appear to be missing 1's for trial_standard_deviation
df['avnr'] = df['trial_standard_deviation'].fillna(1).astype(int)
df.rename({'signal_magnitude': 'cnr'}, axis=1, inplace=True)

#%%
# Draw a categorical scatterplot to show each observation
g = sns.violinplot(x="correlation_target", y="corr_obs_trans_clip",
                   hue="estimation_method",
                   cut=0, data=df)
g.axes.legend(loc='lower right')



#%%
g = sns.violinplot(x="correlation_target", y="corr_obs_trans_clip",
                   hue="estimation_method",
                   cut=0, data=df)
g.axes.legend(loc='lower right')

saf.add_targets(g)

#%%
cnrs = df["cnr"].unique()

#%%

for cnr in cnrs:
    for avnr in df['avnr'].unique():
        df_tmp = df.query(f'(cnr == {cnr}) & (avnr == {avnr})')
        g2 = sns.catplot(x="correlation_target", y="corr_obs_trans_clip",
                         hue="estimation_method", col="iti_mean", row="n_trials",
                         data=df_tmp, kind="violin", cut=0,
                        )
        g2.fig.suptitle("CNR: {}, AVNR: {}".format(cnr, avnr))
        r_targets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        z_targets = [np.arctanh(r) for r in r_targets]
        for sg in g2.axes.ravel():
            saf.add_targets(sg, targets=z_targets)

    save_eps(g2, "../outputs/cnr-{cnr}_avnr-{avnr}_simulations".format(cnr=cnr, avnr=avnr))


#%%
# seminar graph

for cnr in cnrs:
    for avnr in df['avnr'].unique():
        # sns.set_style({'axes.grid': False})
        query = ('(iti_mean == 2.0 | iti_mean == 8.0)'
                 ' & (n_trials == 15 | n_trials == 60)'
                 ' & (correlation_target == 0.2 | correlation_target == 0.4'
                 ' | correlation_target == 0.6 | correlation_target == 0.8)'
                 ' & (cnr == {cnr})'
                 ' & (avnr == {avnr})'.format(cnr=cnr,
                                              avnr=avnr))
        df_tmp = df.query(query)
        tmp_rename = {"correlation_target": "Ground Truth Correlation",
                      "corr_obs_trans_clip": "Observed Correlation (r-z)",
                      "n_trials": "# Events",
                      "iti_mean": "Inter Event Interval",
                      "estimation_method": "Estimation Method"}
        df_tmp.rename(columns=tmp_rename, inplace=True)
        g2 = sns.catplot(x="Ground Truth Correlation", y="Observed Correlation (r-z)",
                         hue="Estimation Method",
                         col="Inter Event Interval", row="# Events",
                         data=df_tmp, kind="violin", cut=0,
                         legend=False, margin_titles=True)

        g2.fig.suptitle("CNR: {}, AVNR: {}".format(cnr, avnr), y=1.02)

        r_targets = [0.2, 0.4, 0.6, 0.8]
        z_targets = [np.arctanh(r) for r in r_targets]
        for sg in g2.axes.ravel():
            y_lab = sg.get_ylabel()
            if y_lab:
                sg.set_ylabel(y_lab, fontdict={'weight': 'heavy'})
            x_lab = sg.get_xlabel()
            if x_lab:
                sg.set_xlabel(x_lab, fontdict={'weight': 'heavy'})

            saf.add_targets(sg, targets=z_targets, width=0.8)

        # add legend
        lgnd_sg = g2.axes[0, 1]
        lgnd_sg.legend(title="Estimation Method",
                    loc="lower left")

        # g2.savefig('../outputs/snr-{snr}_trial_noise-{tng}_simplified_simulation.eps', dpi=600)
        g2.savefig('../outputs/cnr-{cnr}_avnr-{avnr}_simplified_simulation.eps'.format(
            cnr=cnr, avnr=avnr), bbox_inches='tight')


#############################
# Look at false positive rate
#############################
#%%
# test the baseline assumption that no difference correlations
# should have a 5% false positive rate
stat_null_collector = []
pwr_null_collector = {
    "estimator": [],
    "iei_mean": [],
    'n_events': [],
    'cnr': [],
    'avnr': [],
    'power': []
}
for est in ["lsa", "lss"]:
    for iti_mean in [2.0, 4.0, 6.0, 8.0]:
        for n_trials in [15, 30, 45, 60]:
            for cnr in [1, 2]:
                for avnr in [1, 2]:
                    for tgt_corr in [(0.0, 0.0),
                                     (0.1, 0.1),
                                     (0.2, 0.2),
                                     (0.3, 0.3),
                                     (0.4, 0.4),
                                     (0.5, 0.5),
                                     (0.6, 0.6),
                                     (0.7, 0.7),
                                     (0.8, 0.8),
                                     (0.9, 0.9)]:
                        test_df, pwr = saf.test_power(
                            df, estimation_method=est, n_trials=n_trials,
                            iti_mean=iti_mean, signal_magnitude=cnr,
                            correlation_tgt1=tgt_corr[0], correlation_tgt2=tgt_corr[1],
                            simulations=1000, sample_size=50, trial_var=avnr)
                        stat_null_collector.append(test_df)
                        pwr_null_collector['estimator'].append(est)
                        pwr_null_collector['iei_mean'].append(iti_mean)
                        pwr_null_collector['n_events'].append(n_trials)
                        pwr_null_collector['cnr'].append(cnr)
                        pwr_null_collector['avnr'].append(avnr)
                        pwr_null_collector['power'].append(pwr)


#%%
df_null_pwr = pd.concat(stat_null_collector)
pwr_null_summary_df = pd.DataFrame.from_dict(pwr_null_collector)
pwr_null_summary_df = pwr_null_summary_df.groupby(
    ["estimator", 'iei_mean', 'n_events', 'cnr', 'avnr']).describe()[('power', 'mean')].reset_index()
pwr_null_summary_df.columns = pwr_null_summary_df.columns.get_level_values(0)
rename_dict = {
    'estimator': 'Estimator',
    'iei_mean': 'Inter Event Interval (seconds)',
    'n_events': 'Number of Events',
    'cnr': 'CNR',
    'avnr': 'AVNR',
    'power': 'False Positive Rate (%)'}
pwr_null_summary_df.rename(rename_dict, axis=1, inplace=True)

#%%
g = sns.catplot(
    kind='bar', x='Inter Event Interval (seconds)',
    y='False Positive Rate (%)', hue='Estimator',
    col='Number of Events', row='CNR',
    legend=False,
    data=pwr_null_summary_df.query("AVNR==1"))

# set y_axis to have same range
g.set(ylim=(0, 0.06))

# add the red line to indicate 5% false positive rate
g.fig.suptitle("AVNR=1", y=1.02)
for ax in g.axes.ravel():
    ax.axhline(0.05, ls='--', color='red')


# add numbers to each of the bars
saf.show_values_on_bars(g.axes.flatten())

# add a legend (changing title size was *tough*)
g.add_legend(fontsize=30)
g._legend.set_title('Estimator', prop={'size': 20, 'weight': 'heavy'})

save_eps(g.fig, '../outputs/avnr-1_fpr')


#%%
g = sns.catplot(
    kind='bar', x='Inter Event Interval (seconds)',
    y='False Positive Rate (%)', hue='Estimator',
    col='Number of Events', row='CNR', legend=False,
    data=pwr_null_summary_df.query("AVNR==2"))

# set y_axis to have same range
g.set(ylim=(0, 0.06))

g.fig.suptitle("AVNR=2", y=1.02)
for ax in g.axes.ravel():
    ax.axhline(0.05, ls='--', color='red')

# add numbers to each of the bars
saf.show_values_on_bars(g.axes.flatten())

# add a legend (changing title size was *tough*)
g.add_legend(fontsize=30)
g._legend.set_title('Estimator', prop={'size': 20, 'weight': 'heavy'})
# save figure
save_eps(g.fig, '../outputs/avnr-2_fpr')

###############################
# Look at small difference (0.1)
###############################
#%%
stat_null_collector = []
pwr_null_collector = {
    "estimator": [],
    "iei_mean": [],
    'n_events': [],
    'cnr': [],
    'avnr': [],
    'power': []
}
for est in ["lsa", "lss"]:
    for iti_mean in [2.0, 4.0, 6.0, 8.0]:
        for n_trials in [15, 30, 45, 60]:
            for cnr in [1, 2]:
                for avnr in [1, 2]:
                    for tgt_corr in [(0.0, 0.1),
                                     (0.1, 0.2),
                                     (0.2, 0.3),
                                     (0.3, 0.4),
                                     (0.4, 0.5),
                                     (0.5, 0.6),
                                     (0.6, 0.7),
                                     (0.7, 0.8),
                                     (0.8, 0.9)]:
                        test_df, pwr = saf.test_power(
                            df, estimation_method=est, n_trials=n_trials,
                            iti_mean=iti_mean, signal_magnitude=cnr,
                            correlation_tgt1=tgt_corr[0], correlation_tgt2=tgt_corr[1],
                            simulations=1112, sample_size=50, trial_var=avnr)
                        stat_null_collector.append(test_df)
                        pwr_null_collector['estimator'].append(est)
                        pwr_null_collector['iei_mean'].append(iti_mean)
                        pwr_null_collector['n_events'].append(n_trials)
                        pwr_null_collector['cnr'].append(cnr)
                        pwr_null_collector['avnr'].append(avnr)
                        pwr_null_collector['power'].append(pwr)


#%%
df_null_pwr = pd.concat(stat_null_collector)
pwr_null_summary_df = pd.DataFrame.from_dict(pwr_null_collector)
pwr_null_summary_df = pwr_null_summary_df.groupby(
    ["estimator", 'iei_mean', 'n_events', 'cnr', 'avnr']).describe()[('power', 'mean')].reset_index()
pwr_null_summary_df.columns = pwr_null_summary_df.columns.get_level_values(0)
rename_dict = {
    'estimator': 'Estimator',
    'iei_mean': 'Inter Event Interval (seconds)',
    'n_events': 'Number of Events',
    'cnr': 'CNR',
    'avnr': 'AVNR',
    'power': 'Power (% of significant ttests)'}
pwr_null_summary_df.rename(rename_dict, axis=1, inplace=True)

#%%
g = sns.catplot(
    kind='bar', x='Inter Event Interval (seconds)',
    y='Power (% of significant ttests)', hue='Estimator',
    col='Number of Events', row='CNR', legend=False,
    data=pwr_null_summary_df.query("AVNR==1"))

# set y_axis to have same range
g.set(ylim=(0, 1))

g.fig.suptitle("AVNR=1", y=1.02)
for ax in g.axes.ravel():
    ax.axhline(0.80, ls='--', color='red')

# add numbers to each of the bars
saf.show_values_on_bars(g.axes.flatten())

# add a legend (changing title size was *tough*)
g.add_legend(fontsize=30)
g._legend.set_title('Estimator', prop={'size': 20, 'weight': 'heavy'})

save_eps(g.fig, '../outputs/avnr-1_smalldiff')

#%%
g = sns.catplot(
    kind='bar', x='Inter Event Interval (seconds)',
    y='Power (% of significant ttests)', hue='Estimator',
    col='Number of Events', row='CNR', legend=False,
    data=pwr_null_summary_df.query("AVNR==2"))

# set y_axis to have same range
g.set(ylim=(0, 1))

g.fig.suptitle("AVNR=2", y=1.02)
for ax in g.axes.ravel():
    ax.axhline(0.80, ls='--', color='red')

# add numbers to each of the bars
saf.show_values_on_bars(g.axes.flatten())

# add a legend (changing title size was *tough*)
g.add_legend(fontsize=30)
g._legend.set_title('Estimator', prop={'size': 20, 'weight': 'heavy'})

save_eps(g.fig, '../outputs/avnr-2_smalldiff')

################################
# Look at large difference (0.3)
################################
#%%
stat_null_collector = []
pwr_null_collector = {
    "estimator": [],
    "iei_mean": [],
    'n_events': [],
    'cnr': [],
    'avnr': [],
    'power': []
}
for est in ["lsa", "lss"]:
    for iti_mean in [2.0, 4.0, 6.0, 8.0]:
        for n_trials in [15, 30, 45, 60]:
            for cnr in [1, 2]:
                for avnr in [1, 2]:
                    for tgt_corr in [(0.0, 0.3),
                                     (0.1, 0.4),
                                     (0.2, 0.5),
                                     (0.3, 0.6),
                                     (0.4, 0.7),
                                     (0.5, 0.8),
                                     (0.6, 0.9)]:
                        test_df, pwr = saf.test_power(
                            df, estimation_method=est, n_trials=n_trials,
                            iti_mean=iti_mean, signal_magnitude=cnr,
                            correlation_tgt1=tgt_corr[0], correlation_tgt2=tgt_corr[1],
                            simulations=1429, sample_size=50, trial_var=avnr)
                        stat_null_collector.append(test_df)
                        pwr_null_collector['estimator'].append(est)
                        pwr_null_collector['iei_mean'].append(iti_mean)
                        pwr_null_collector['n_events'].append(n_trials)
                        pwr_null_collector['cnr'].append(cnr)
                        pwr_null_collector['avnr'].append(avnr)
                        pwr_null_collector['power'].append(pwr)


#%%
df_null_pwr = pd.concat(stat_null_collector)
pwr_null_summary_df = pd.DataFrame.from_dict(pwr_null_collector)
pwr_null_summary_df = pwr_null_summary_df.groupby(
    ["estimator", 'iei_mean', 'n_events', 'cnr', 'avnr']).describe()[('power', 'mean')].reset_index()
pwr_null_summary_df.columns = pwr_null_summary_df.columns.get_level_values(0)
rename_dict = {
    'estimator': 'Estimator',
    'iei_mean': 'Inter Event Interval (seconds)',
    'n_events': 'Number of Events',
    'cnr': 'CNR',
    'avnr': 'AVNR',
    'power': 'Power (% of significant ttests)'}
pwr_null_summary_df.rename(rename_dict, axis=1, inplace=True)

#%%
g = sns.catplot(
    kind='bar', x='Inter Event Interval (seconds)',
    y='Power (% of significant ttests)', hue='Estimator',
    col='Number of Events', row='CNR', legend=False,
    data=pwr_null_summary_df.query("AVNR==1"))

# set y_axis to have same range
g.set(ylim=(0, 1))

g.fig.suptitle("AVNR=1", y=1.02)
for ax in g.axes.ravel():
    ax.axhline(0.80, ls='--', color='red')

# add numbers to each of the bars
saf.show_values_on_bars(g.axes.flatten())

# add a legend (changing title size was *tough*)
g.add_legend(fontsize=30)
g._legend.set_title('Estimator', prop={'size': 20, 'weight': 'heavy'})
g.fig.savefig('../outputs/avnr-1_largediff.eps', bbox_inches='tight')

#%%
g = sns.catplot(
    kind='bar', x='Inter Event Interval (seconds)',
    y='Power (% of significant ttests)', hue='Estimator',
    col='Number of Events', row='CNR', legend=False,
    data=pwr_null_summary_df.query("AVNR==2"))

# set y_axis to have same range
g.set(ylim=(0, 1))

g.fig.suptitle("AVNR=10", y=1.02)
for ax in g.axes.ravel():
    ax.axhline(0.80, ls='--', color='red')

# add numbers to each of the bars
saf.show_values_on_bars(g.axes.flatten())

# add a legend (changing title size was *tough*)
g.add_legend(fontsize=30)
g._legend.set_title('Estimator', prop={'size': 20, 'weight': 'heavy'})
g.fig.savefig('../outputs/avnr-2_largediff.eps', bbox_inches='tight')

########################
# SIMULATE ANALYZED TASK
########################
#%% Simulated data using a real task (task switch)
df = pd.read_csv('../simulation_3000_task_switch.tsv', sep='\t')

#%%
df.loc[df['correlation_observed'] == 1, 'correlation_observed'] -= 0.0001
df.loc[df['correlation_observed'] == -1, 'correlation_observed'] += 0.0001
df['corr_obs_trans'] = np.arctanh(df['correlation_observed'])
df['corr_obs_trans_clip'] = np.clip(df['corr_obs_trans'], -1.5, 1.5)
# Some simulations appear to be missing 1's for trial_standard_deviation
df['avnr'] = df['trial_standard_deviation'].fillna(1).astype(int)
df.rename({'signal_magnitude': 'cnr'}, axis=1, inplace=True)

#%%
# test the baseline assumption that no difference correlations
# should have a 5% false positive rate
stat_null_collector = []
pwr_null_collector = {
    "estimator": [],
    'cnr': [],
    'avnr': [],
    'power': [],
}
for est in ["lsa", "lss"]:
    for cnr in [1, 2]:
        for avnr in [1, 2]:
            for tgt_corr in [(0.0, 0.0),
                             (0.1, 0.1),
                             (0.2, 0.2),
                             (0.3, 0.3),
                             (0.4, 0.4),
                             (0.5, 0.5),
                             (0.6, 0.6),
                             (0.7, 0.7),
                             (0.8, 0.8),
                             (0.9, 0.9)]:
                test_df, pwr = saf.test_power(
                    df, estimation_method=est, n_trials=50,
                    iti_mean=18.84, signal_magnitude=cnr,
                    correlation_tgt1=tgt_corr[0], correlation_tgt2=tgt_corr[1],
                    trial_type1='repeat', trial_type2='switch',
                    simulations=1000, sample_size=40, trial_var=avnr)
                stat_null_collector.append(test_df)
                pwr_null_collector['estimator'].append(est)
                pwr_null_collector['cnr'].append(cnr)
                pwr_null_collector['avnr'].append(avnr)
                pwr_null_collector['power'].append(pwr)

#%%
df_null_pwr = pd.concat(stat_null_collector)
pwr_null_summary_df = pd.DataFrame.from_dict(pwr_null_collector)
pwr_null_summary_df = pwr_null_summary_df.groupby(
    ["estimator", 'cnr', 'avnr']).describe()[('power', 'mean')].reset_index()
pwr_null_summary_df.columns = pwr_null_summary_df.columns.get_level_values(0)
rename_dict = {
    'estimator': 'Estimator',
    'cnr': 'CNR',
    'avnr': 'AVNR',
    'power': 'False Positive Rate (%)'}
pwr_null_summary_df.rename(rename_dict, axis=1, inplace=True)

#%%
g = sns.catplot(
    kind='bar', x='CNR',
    y='False Positive Rate (%)', hue='Estimator',
    col='AVNR', legend=False,
    data=pwr_null_summary_df)

# set y_axis to have same range
g.set(ylim=(0, 0.06))

for ax in g.axes.ravel():
    ax.axhline(0.05, ls='--', color='red')

# add numbers to each of the bars
saf.show_values_on_bars(g.axes.flatten())

# add a legend (changing title size was *tough*)
g.add_legend(fontsize=30)
g._legend.set_title('Estimator', prop={'size': 20, 'weight': 'heavy'})

save_eps(g.fig, '../outputs/taskswitch-switchXrepeat_fpr')


#%%
stat_null_collector = []
pwr_null_collector = {
    "estimator": [],
    'cnr': [],
    'avnr': [],
    'power': [],
}
for est in ["lsa", "lss"]:
    for cnr in [1, 2]:
        for avnr in [1, 2]:
            for tgt_corr in [(0.0, 0.0),
                             (0.1, 0.1),
                             (0.2, 0.2),
                             (0.3, 0.3),
                             (0.4, 0.4),
                             (0.5, 0.5),
                             (0.6, 0.6),
                             (0.7, 0.7),
                             (0.8, 0.8),
                             (0.9, 0.9)]:
                test_df, pwr = saf.test_power(
                    df, estimation_method=est, n_trials=50,
                    iti_mean=18.84, signal_magnitude=cnr,
                    correlation_tgt1=tgt_corr[0], correlation_tgt2=tgt_corr[1],
                    trial_type1='single', trial_type2='switch',
                    simulations=1000, sample_size=40, trial_var=avnr)
                stat_null_collector.append(test_df)
                pwr_null_collector['estimator'].append(est)
                pwr_null_collector['cnr'].append(cnr)
                pwr_null_collector['avnr'].append(avnr)
                pwr_null_collector['power'].append(pwr)

#%%
df_null_pwr = pd.concat(stat_null_collector)
pwr_null_summary_df = pd.DataFrame.from_dict(pwr_null_collector)
pwr_null_summary_df = pwr_null_summary_df.groupby(
    ["estimator", 'cnr', 'avnr']).describe()[('power', 'mean')].reset_index()
pwr_null_summary_df.columns = pwr_null_summary_df.columns.get_level_values(0)
rename_dict = {
    'estimator': 'Estimator',
    'cnr': 'CNR',
    'avnr': 'AVNR',
    'power': 'False Positive Rate (%)'}
pwr_null_summary_df.rename(rename_dict, axis=1, inplace=True)

#%%
g = sns.catplot(
    kind='bar', x='CNR',
    y='False Positive Rate (%)', hue='Estimator',
    col='AVNR', legend=False,
    data=pwr_null_summary_df)

# set y_axis to have same range
g.set(ylim=(0, 0.06))

for ax in g.axes.ravel():
    ax.axhline(0.05, ls='--', color='red')

# add numbers to each of the bars
saf.show_values_on_bars(g.axes.flatten())

# add a legend (changing title size was *tough*)
g.add_legend(fontsize=30)
g._legend.set_title('Estimator', prop={'size': 20, 'weight': 'heavy'})

save_eps(g.fig, '../outputs/taskswitch-switchXsingle_fpr')

###############################
# Look at small difference (0.1)
###############################
#%%
# test the baseline assumption that no difference correlations
# should have a 5% false positive rate
stat_collector = []
pwr_null_collector = {
    "estimator": [],
    'cnr': [],
    'avnr': [],
    'power': [],
}
for est in ["lsa", "lss"]:
    for cnr in [1, 2]:
        for avnr in [1, 2]:
            for tgt_corr in [(0.0, 0.1),
                             (0.1, 0.2),
                             (0.2, 0.3),
                             (0.3, 0.4),
                             (0.4, 0.5),
                             (0.5, 0.6),
                             (0.6, 0.7),
                             (0.7, 0.8),
                             (0.8, 0.9)]:
                test_df, pwr = saf.test_power(
                    df, estimation_method=est, n_trials=50,
                    iti_mean=18.84, signal_magnitude=cnr,
                    correlation_tgt1=tgt_corr[0], correlation_tgt2=tgt_corr[1],
                    trial_type1='repeat', trial_type2='switch',
                    simulations=1112, sample_size=60, trial_var=avnr)
                stat_null_collector.append(test_df)
                pwr_null_collector['estimator'].append(est)
                pwr_null_collector['cnr'].append(cnr)
                pwr_null_collector['avnr'].append(avnr)
                pwr_null_collector['power'].append(pwr)

#%%
df_null_pwr = pd.concat(stat_null_collector)
pwr_null_summary_df = pd.DataFrame.from_dict(pwr_null_collector)
pwr_null_summary_df = pwr_null_summary_df.groupby(
    ["estimator", 'cnr', 'avnr']).describe()[('power', 'mean')].reset_index()
pwr_null_summary_df.columns = pwr_null_summary_df.columns.get_level_values(0)
rename_dict = {
    'estimator': 'Estimator',
    'cnr': 'CNR',
    'avnr': 'AVNR',
    'power': 'Power (% of significant ttests)',
}
pwr_null_summary_df.rename(rename_dict, axis=1, inplace=True)

#%%
g = sns.catplot(
    kind='bar', x='CNR',
    y='Power (% of significant ttests)', hue='Estimator',
    col='AVNR', legend=False,
    data=pwr_null_summary_df)

# set y_axis to have same range
g.set(ylim=(0, 1))

for ax in g.axes.ravel():
    ax.axhline(0.80, ls='--', color='red')

# add numbers to each of the bars
saf.show_values_on_bars(g.axes.flatten())

# add a legend (changing title size was *tough*)
g.add_legend(fontsize=30)
g._legend.set_title('Estimator', prop={'size': 20, 'weight': 'heavy'})

save_eps(g.fig, '../outputs/taskswitch-switchXrepeat_smalldiff')

###############################
# Look at large difference (0.3)
###############################
#%%
stat_null_collector = []
pwr_null_collector = {
    "estimator": [],
    'cnr': [],
    'avnr': [],
    'power': [],
}
for est in ["lsa", "lss"]:
    for cnr in [1, 2]:
        for avnr in [1, 2]:
            for tgt_corr in [(0.0, 0.3),
                             (0.1, 0.4),
                             (0.2, 0.5),
                             (0.3, 0.6),
                             (0.4, 0.7),
                             (0.5, 0.8),
                             (0.6, 0.9)]:
                test_df, pwr = saf.test_power(
                    df, estimation_method=est, n_trials=50,
                    iti_mean=18.84, signal_magnitude=cnr,
                    correlation_tgt1=tgt_corr[0], correlation_tgt2=tgt_corr[1],
                    trial_type1='repeat', trial_type2='switch',
                    simulations=1429, sample_size=40, trial_var=avnr)
                stat_null_collector.append(test_df)
                pwr_null_collector['estimator'].append(est)
                pwr_null_collector['cnr'].append(cnr)
                pwr_null_collector['avnr'].append(avnr)
                pwr_null_collector['power'].append(pwr)

#%%
df_null_pwr = pd.concat(stat_null_collector)
pwr_null_summary_df = pd.DataFrame.from_dict(pwr_null_collector)
pwr_null_summary_df = pwr_null_summary_df.groupby(
    ["estimator", 'cnr', 'avnr']).describe()[('power', 'mean')].reset_index()
pwr_null_summary_df.columns = pwr_null_summary_df.columns.get_level_values(0)
rename_dict = {
    'estimator': 'Estimator',
    'cnr': 'CNR',
    'avnr': 'AVNR',
    'power': 'Power (% of significant ttests)',
}
pwr_null_summary_df.rename(rename_dict, axis=1, inplace=True)

#%%
g = sns.catplot(
    kind='bar', x='CNR',
    y='Power (% of significant ttests)', hue='Estimator',
    col='AVNR', legend=False,
    data=pwr_null_summary_df)

# set y_axis to have same range
g.set(ylim=(0, 1))

for ax in g.axes.ravel():
    ax.axhline(0.80, ls='--', color='red')

# add numbers to each of the bars
saf.show_values_on_bars(g.axes.flatten())

# add a legend (changing title size was *tough*)
g.add_legend(fontsize=30)
g._legend.set_title('Estimator', prop={'size': 20, 'weight': 'heavy'})

g.fig.savefig('../outputs/taskswitch-switchXrepeat_largediff.eps', bbox_inches='tight')

#########
# How many participants are required for 80% power?
#########
#%%
bold_pwr_dict = {
    "method": [],
    "power": [],
    "participants": [],
    "cnr": [],
    "avnr": [],
    }
for method in ["lsa", "lss"]:
    # from 5 participants to 40
    for participants in range(5, 61):
        for tgt_corr in [(0.0, 0.1),
                         (0.1, 0.2),
                         (0.2, 0.3),
                         (0.3, 0.4),
                         (0.4, 0.5),
                         (0.5, 0.6),
                         (0.6, 0.7),
                         (0.7, 0.8),
                         (0.8, 0.9)]:
            bold_test_df, bold_pwr = saf.test_power(
                df,
                estimation_method=method,
                iti_mean=18.84,
                n_trials=50,
                signal_magnitude=2,
                trial_var=2,
                correlation_tgt1=tgt_corr[0],
                correlation_tgt2=tgt_corr[1],
                trial_type1='repeat',
                trial_type2='switch',
                sample_size=participants,
                simulations=1112)
            bold_pwr_dict['method'].append(method)
            bold_pwr_dict['participants'].append(participants)
            bold_pwr_dict['power'].append(bold_pwr)
            bold_pwr_dict['cnr'].append(2)
            bold_pwr_dict['avnr'].append(2)

#%%
bold_pwr_df = pd.DataFrame.from_dict(bold_pwr_dict)
bold_pwr_df['power'] = bold_pwr_df['power'] * 100
bold_pwr_df.head()

#%%
bold_pwr_df['power']
fig, ax = plt.subplots()
sns.lineplot(
    x='participants',
    y='power',
    hue='method',
    legend="brief",
    hue_order=["lsa", "lss"],
    ax=ax,
    err_style=None,
    data=bold_pwr_df)

ax.axhline(80, ls='--', color='red')

lss_line = ax.get_lines()[0]
lsa_line = ax.get_lines()[1]
lss_line.set_linewidth(4)
lsa_line.set_linewidth(4)
lss_80_power_idx = np.argmax(lss_line.get_data()[1] > 80)
lss_80_power = lss_line.get_data()[0][lss_80_power_idx]

ax.axvline(lss_80_power, ls='--', color=lss_line.get_color())
ax.set_xlim(5, 60)
ax.set_xlabel("# of Participants", weight="heavy")
ax.set_ylabel("Power (%)", weight="heavy")

ax.annotate("80% Power", (30, 81))

save_eps(fig, '../outputs/taskswitch-switchXrepeat_smalldiffparticipants')


############
# PLAYGROUND
############
# #%%
# tmp_rename['p_value'] = "p value"
# df_pwr_tmp = df_pwr.rename(columns=tmp_rename)


# def distplot_plus(x, sig_mag, trial_var, **kwargs):
#     txtstr = dedent(r"""lss pwr: {lss_pwr}
# lsa pwr: {lsa_pwr}""")
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     ax = sns.distplot(x, **kwargs)
#     ax.set_yticklabels([])
#     lss_pwr = np.array(pwr_collector[("lss", sig_mag.unique()[0], trial_var.unique()[0])]).mean()
#     lsa_pwr = np.array(pwr_collector[("lsa", sig_mag.unique()[0], trial_var.unique()[0])]).mean()

#     txtrepl = txtstr.format(lss_pwr=np.round(lss_pwr, 2),
#                             lsa_pwr=np.round(lsa_pwr, 2))
#     ax.text(0.25, 0.45, txtrepl, transform=ax.transAxes, fontsize=14,
#             verticalalignment='center',
#             horizontalalignment="left", bbox=props)
#     return ax

# g_fac = sns.FacetGrid(df_pwr_tmp, row="signal_magnitude", col="trial_var",
#                     hue="Estimation Method",
#                     hue_order=["lss", "lsa"], margin_titles=True,
#                     sharey=False)

# g_hist = (g_fac.map(distplot_plus, "p value", "signal_magnitude", "trial_var", hist=True, bins=20))
# sg_hist_lgnd = g_hist.axes[0, 1]
# sg_hist_lgnd.legend()


# # g_hist.savefig('../outputs/snr-{}_trial_noise-{}_simplified_pwr.eps', dpi=400)
# g_hist.savefig('../outputs/diff-small_task-taskswitch_simplified_pwr.eps')

# #%%
# # use real data
# events_df = pd.read_csv(
#     'data/test_bold/bids/sub-GE120012/ses-pre/func/sub-GE120012_ses-pre_task-taskswitch_events.tsv', sep='\t'
# )


# #%%
# if "switch" in events_df.columns:
#     events_df.loc[:, "switch"].replace({0: "single",
#                                         9: "single",
#                                         1: "repeat",
#                                         2: "switch"},
#                                        inplace=True)
#     events_df.rename(
#         columns={"trial_type": "stim_color",
#                  "switch": "trial_type"},
#         inplace=True)

# #%%
# events_df.to_csv('data/test_bold/mod.tsv', sep='\t', index=False)


# #%%
# fname_template = os.path.join(
#     "data",
#     "test_bold",
#     "bids",
#     "derivatives",
#     "nibetaseries",
#     "sub-GE120012",
#     "ses-pre",
#     "func",
#     "sub-GE120012_ses-pre_task-taskswitch_space-MNI152NLin2009cAsym_desc-{cond}_correlation.tsv"
# )


# def rename_index(x):
#     res = x.split('_')
#     network_index = res.pop(1)
#     roi_index = '_'.join(res)

#     return network_index, roi_index


# df_dict = {}
# for cond in ["switch", "repeat", "single"]:
#     df_dict[cond] = pd.read_csv(
#         fname_template.format(cond=cond),
#         sep="\t", index_col=0)
#     df_dict[cond].index = pd.MultiIndex.from_tuples(
#         list(df_dict[cond].index.map(rename_index)))
#     df_dict[cond].columns = pd.MultiIndex.from_tuples(
#         list(df_dict[cond].columns.map(rename_index)))
#     df_dict[cond].sort_index(axis=0, inplace=True)
#     df_dict[cond].sort_index(axis=1, inplace=True)
#     df_dict[cond] = df_dict[cond].mean(level=0, axis=0).mean(level=0, axis=1)


# #%%
# sns.heatmap(df_dict['switch'] - df_dict['repeat'])

# #%%
# for cond in ["switch", "repeat", "single"]:
#     sns.heatmap(df_dict[cond], vmax=1.2)
#     plt.show()

# #%%
# df_dict[cond].describe()

# #%%
# df_sub = df_dict['switch'] - df_dict['repeat']
# ax = sns.heatmap(
#     df_sub,
#     cmap='viridis',
#     )

# ax.set_title(
#     label="Switch - Repeat",
#     fontdict={"fontsize": 20, "fontweight": "heavy"})
# cbar = ax.collections[0].colorbar
# cbar.set_label('Correlation Difference\n(Fisher r-z)',
#                fontsize=15,
#                fontweight="heavy")
# cbar.ax.tick_params(labelsize=13)

# ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
# plt.tight_layout()
# ax.figure.savefig("outputs/beta_series_contrast_switch-repeat.eps")
# ax.figure.savefig("outputs/beta_series_contrast_switch-repeat.eps", dpi=400)
# #%%
# # get the bold simulation results
# bold_sim_file = os.path.join(
#     "data",
#     "test_bold",
#     "bold_simulation.tsv")
# df_bold_sim = pd.read_csv(bold_sim_file, sep="\t")
# df_bold_sim.head()

# #%%
# df_bold_sim.loc[df_bold_sim['correlation_observed'] == 1, 'correlation_observed'] -= 0.0001
# df_bold_sim.loc[df_bold_sim['correlation_observed'] == -1, 'correlation_observed'] += 0.0001
# df_bold_sim['corr_obs_trans'] = np.arctanh(df_bold_sim['correlation_observed'])
# df_bold_sim['corr_obs_trans_clip'] = np.clip(df_bold_sim['corr_obs_trans'], -1.5, 1.5)

# #%%
# bold_pwr_dict = {"method": [], "power": [], "participants": []}
# for method in ["lsa", "lss"]:
#     # from 5 participants to 40
#     for participants in range(5, 41):
#         bold_test_df, bold_pwr = test_power(
#             df_bold_sim,
#             estimation_method=method,
#             iti_mean=18.84,
#             n_trials=50,
#             signal_magnitude=1.7502511308736108,
#             correlation_tgt1=0.0,
#             correlation_tgt2=0.1,
#             sample_size=participants)
#         bold_pwr_dict['method'].append(method)
#         bold_pwr_dict['participants'].append(participants)
#         bold_pwr_dict['power'].append(bold_pwr)

# bold_pwr_df = pd.DataFrame.from_dict(bold_pwr_dict)
# bold_pwr_df.head()

# #%%
# pp = sns.lineplot(
#     x='participants',
#     y='power',
#     hue='method',
#     legend="brief",
#     hue_order=["lss", "lsa"],
#     data=bold_pwr_df)

# lgnd = pp.legend()
# lgnd.get_texts()[0].set_text('')
# for txt in lgnd.get_texts()[1:]:
#     txt.set_fontsize(15)
# lgnd.set_title("Estimation Method", prop={'size': 15, 'weight': 'heavy'})
# pp.set_xlabel("# of Participants",
#               fontdict={'fontsize': 15, 'fontweight': 'heavy'})
# pp.set_ylabel("Power",
#               fontdict={'fontsize': 15, 'fontweight': 'heavy'})
# pp.axhline(0.8, linestyle='--')
# plt.tight_layout()
# pp.figure.savefig('outputs/power_plot.eps')
# pp.figure.savefig('outputs/power_plot.eps', dpi=400)

# #%%
# stat_collector = []
# pwr_collector = {}
# for est in ["lsa", "lss"]:
#     test_df, pwr = test_power(
#                     df, estimation_method=est, n_trials=50,
#                     iti_mean=18.84, signal_magnitude=1,
#                     correlation_tgt1=0.1, correlation_tgt2=0.4,
#                     trial_type1="repeat", trial_type2="switch",
#                     sample_size=30, trial_var=1)
#     stat_collector.append(test_df)
#     pwr_collector[est] = pwr


# # %%
# tips = sns.load_dataset("tips")
# from scipy import stats
# def qqplot(x, y, **kwargs):
#     _, xr = stats.probplot(x, fit=False)
#     _, yr = stats.probplot(y, fit=False)
#     plt.scatter(xr, yr, **kwargs)

# g = sns.FacetGrid(tips, col="smoker", height=4)
# g.map(qqplot, "total_bill", "tip");

# # %%


# %%
