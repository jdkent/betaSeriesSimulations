
#%%
import os
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_1samp
sns.set(style="ticks", rc={"lines.linewidth": 3.0})
sns.set(font_scale=1.3)
sns.set_style({'axes.grid': False})
plt.style.use("analysis/gray_background.mplstyle")

#%%
df = pd.read_csv('data/bulk_simulation32.tsv', sep='\t')

#%%
df.loc[df['correlation_observed'] == 1, 'correlation_observed'] -= 0.0001
df.loc[df['correlation_observed'] == -1, 'correlation_observed'] += 0.0001
df['corr_obs_trans'] = np.arctanh(df['correlation_observed'])
df['corr_obs_trans_clip'] = np.clip(df['corr_obs_trans'], -1.5, 1.5)
#%%
# Draw a categorical scatterplot to show each observation
g = sns.violinplot(x="correlation_target", y="corr_obs_trans",
                   hue="estimation_method",
                   cut=0, data=df)
g.axes.legend(loc='lower right')

#%%
def add_targets(g, bound=0.01, width=0.5, targets=None):
    if targets is None:
        targets = [float(x.get_text()) for x in g.get_xticklabels()]
    for idx, target in enumerate(targets):
        trans = g.transData
        g.add_patch(Rectangle((idx - (width/2), target - bound),
                              width, bound*2, color='red',
                              zorder=100, transform=trans))

    return g


#%%
g = sns.violinplot(x="correlation_target", y="corr_obs_trans_clip",
                   hue="estimation_method",
                   cut=0, data=df)
g.axes.legend(loc='lower right')

add_targets(g)

#%%
snrs = df["signal_magnitude"].unique()

#%%

for snr in snrs:
    df_tmp = df[df["signal_magnitude"] == snr]
    g2 = sns.catplot(x="correlation_target", y="corr_obs_trans_clip",
                     hue="estimation_method", col="iti_mean", row="n_trials",
                     data=df_tmp, kind="violin", cut=0,
                     )
    g2.fig.suptitle("CNR: {}".format(snr))
    r_targets = [-0.8, 0.0, 0.2, 0.4, 0.6, 0.8]
    z_targets = [np.arctanh(r) for r in r_targets]
    for sg in g2.axes.ravel():
        add_targets(sg, targets=z_targets)

    g2.savefig("./outputs/snr-{snr}_simulations.svg".format(snr=snr))


#%%
# seminar graph
# graph 37.06 CNR
snr = snrs[1]

# sns.set_style({'axes.grid': False})
query = ('(iti_mean == 2.0 | iti_mean == 8.0)'
         ' & (n_trials == 15 | n_trials == 60)'
         ' & (correlation_target == 0.2 | correlation_target == 0.4'
         ' | correlation_target == 0.6 | correlation_target == 0.8)'
         ' & (signal_magnitude == {snr})'.format(snr=snr))
df_tmp = df.query(query)
tmp_rename = {"correlation_target": "Ground Truth Correlation",
              "corr_obs_trans_clip": "Observed Correlation (r-z)",
              "n_trials": "# Trials",
              "iti_mean": "Inter Trial Interval",
              "estimation_method": "Estimation Method"}
df_tmp.rename(columns=tmp_rename, inplace=True)
g2 = sns.catplot(x="Ground Truth Correlation", y="Observed Correlation (r-z)",
                 hue="Estimation Method",
                 col="Inter Trial Interval", row="# Trials",
                 data=df_tmp, kind="violin", cut=0,
                 legend=False, margin_titles=True)

r_targets = [0.2, 0.4, 0.6, 0.8]
z_targets = [np.arctanh(r) for r in r_targets]
for sg in g2.axes.ravel():
    y_lab = sg.get_ylabel()
    if y_lab:
        sg.set_ylabel(y_lab, fontdict={'weight': 'heavy'})
    x_lab = sg.get_xlabel()
    if x_lab:
        sg.set_xlabel(x_lab, fontdict={'weight': 'heavy'})

    add_targets(sg, targets=z_targets, width=0.8)

# add legend
lgnd_sg = g2.axes[0, 1]
lgnd_sg.legend(title="Estimation Method",
               loc="lower left")

g2.savefig('outputs/simplified_simulation.png', dpi=600)
g2.savefig('outputs/simplified_simulation.svg')

#%%
# test the null (appears to hold up within nominal error rates)
short_lsa_query = ('(estimation_method == "lsa") &'
                   '(iti_mean == 2.0) &'
                   '(n_trials == 15) &'
                   '(correlation_target == 0.4) &'
                   '(signal_magnitude == 37.06)')
df_short_lsa = df.query(short_lsa_query)
test_collector = {"t": [], "p": []}
for _ in range(10000):
    sample = np.random.choice(df_short_lsa['corr_obs_trans'], size=(40, 2),
                              replace=False)
    group1 = sample[:, 0]
    group2 = sample[:, 1]
    t, p = ttest_ind(group1, group2)
    test_collector["t"].append(t)
    test_collector["p"].append(p)

test_df = pd.DataFrame.from_dict(test_collector)

sns.distplot(test_df["p"], bins=20, kde=False)

#%%
df_short_lsa.query('iteration == 0')

#%%
def test_power(df, estimation_method="lss", iti_mean=6.0,
               n_trials=30, correlation_tgt1=0.4, correlation_tgt2=0.6,
               signal_magnitude=37.06, simulations=10000,
               sample_size=40):
    base_query = ('(estimation_method == "{em}") &'
                  '(iti_mean == {iti}) &'
                  '(n_trials == {trl}) &'
                  '(signal_magnitude == {snr})').format(
                    em=estimation_method,
                    iti=iti_mean,
                    trl=n_trials,
                    snr=signal_magnitude
                  )
    group1_query = base_query + \
        '& (correlation_target == {tgt})'.format(
            tgt=correlation_tgt1)

    group2_query = base_query + \
        '& (correlation_target == {tgt})'.format(
            tgt=correlation_tgt2)

    group1 = df.query(group1_query)
    group2 = df.query(group2_query)
    group_diff = group1['corr_obs_trans'].values - group2['corr_obs_trans'].values

    test_collector = {"t_value": [], "p_value": [],
                      "estimation_method": [estimation_method] * simulations,
                      "iti_mean": [iti_mean] * simulations,
                      "n_trials": [n_trials] * simulations,
                      "signal_magnitude": [signal_magnitude] * simulations}

    for _ in range(simulations):
        sample = np.random.choice(group_diff, size=sample_size,
                                  replace=False)
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


#%%
stat_collector = []
pwr_collector = {}
for est in ["lsa", "lss"]:
    for iti_mean in [2.0, 8.0]:
        for n_trials in [15, 60]:
            test_df, pwr = test_power(
                df, estimation_method=est, n_trials=n_trials,
                iti_mean=iti_mean)
            stat_collector.append(test_df)
            pwr_collector[(est, iti_mean, n_trials)] = pwr
#%%
df_pwr = pd.concat(stat_collector)


#%%
tmp_rename['p_value'] = "p value"
df_pwr_tmp = df_pwr.rename(columns=tmp_rename)

g_fac = sns.FacetGrid(df_pwr_tmp, row="# Trials", col="Inter Trial Interval",
                      hue="Estimation Method",
                      hue_order=["lss", "lsa"], margin_titles=True,
                      sharey=False)

g_hist = (g_fac.map(sns.distplot, "p value", hist=True, bins=20))
sg_hist_lgnd = g_hist.axes[0, 1]
sg_hist_lgnd.legend()

txtstr = r"""lss pwr: {lss_pwr}
lsa pwr: {lsa_pwr}"""


props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
for iti_mean, ax, n_trials in zip([2.0, 8.0, 2.0, 8.0],
                                  g_hist.axes.ravel(),
                                  [15, 15, 60, 60]):
    lss_pwr = pwr_collector[("lss", iti_mean, n_trials)]
    lsa_pwr = pwr_collector[("lsa", iti_mean, n_trials)]

    txtrepl = txtstr.format(lss_pwr=np.round(lss_pwr, 2),
                            lsa_pwr=np.round(lsa_pwr, 2))
    ax.text(0.25, 0.45, txtrepl, transform=ax.transAxes, fontsize=14,
            verticalalignment='center',
            horizontalalignment="left", bbox=props)
g_hist.savefig('outputs/simplified_pwr.png', dpi=400)
g_hist.savefig('outputs/simplified_pwr.svg')

# TEST ON REAL DATA
## get the events.tsv formatted correctly
#%%
# use real data
events_df = pd.read_csv(
    'data/test_bold/bids/sub-GE120012/ses-pre/func/sub-GE120012_ses-pre_task-taskswitch_events.tsv', sep='\t'
)


#%%
if "switch" in events_df.columns:
    events_df.loc[:, "switch"].replace({0: "single",
                                        9: "single",
                                        1: "repeat",
                                        2: "switch"},
                                       inplace=True)
    events_df.rename(
        columns={"trial_type": "stim_color",
                 "switch": "trial_type"},
        inplace=True)

#%%
events_df.to_csv('data/test_bold/mod.tsv', sep='\t', index=False)


#%%
fname_template = os.path.join(
    "data",
    "test_bold",
    "bids",
    "derivatives",
    "nibetaseries",
    "sub-GE120012",
    "ses-pre",
    "func",
    "sub-GE120012_ses-pre_task-taskswitch_space-MNI152NLin2009cAsym_desc-{cond}_correlation.tsv"
)


def rename_index(x):
    res = x.split('_')
    network_index = res.pop(1)
    roi_index = '_'.join(res)

    return network_index, roi_index


df_dict = {}
for cond in ["switch", "repeat", "single"]:
    df_dict[cond] = pd.read_csv(
        fname_template.format(cond=cond),
        sep="\t", index_col=0)
    df_dict[cond].index = pd.MultiIndex.from_tuples(
        list(df_dict[cond].index.map(rename_index)))
    df_dict[cond].columns = pd.MultiIndex.from_tuples(
        list(df_dict[cond].columns.map(rename_index)))
    df_dict[cond].sort_index(axis=0, inplace=True)
    df_dict[cond].sort_index(axis=1, inplace=True)
    df_dict[cond] = df_dict[cond].mean(level=0, axis=0).mean(level=0, axis=1)


#%%
sns.heatmap(df_dict['switch'] - df_dict['repeat'])

#%%
for cond in ["switch", "repeat", "single"]:
    sns.heatmap(df_dict[cond], vmax=1.2)
    plt.show()

#%%
df_dict[cond].describe()

#%%
df_sub = df_dict['switch'] - df_dict['repeat']
ax = sns.heatmap(
    df_sub,
    cmap='viridis',
    )

ax.set_title(
    label="Switch - Repeat",
    fontdict={"fontsize": 20, "fontweight": "heavy"})
cbar = ax.collections[0].colorbar
cbar.set_label('Correlation Difference\n(Fisher r-z)',
               fontsize=15,
               fontweight="heavy")
cbar.ax.tick_params(labelsize=13)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
plt.tight_layout()
ax.figure.savefig("outputs/beta_series_contrast_switch-repeat.svg")
ax.figure.savefig("outputs/beta_series_contrast_switch-repeat.png", dpi=400)
#%%
# get the bold simulation results
bold_sim_file = os.path.join(
    "data",
    "test_bold",
    "bold_simulation.tsv")
df_bold_sim = pd.read_csv(bold_sim_file, sep="\t")
df_bold_sim.head()

#%%
df_bold_sim.loc[df_bold_sim['correlation_observed'] == 1, 'correlation_observed'] -= 0.0001
df_bold_sim.loc[df_bold_sim['correlation_observed'] == -1, 'correlation_observed'] += 0.0001
df_bold_sim['corr_obs_trans'] = np.arctanh(df_bold_sim['correlation_observed'])
df_bold_sim['corr_obs_trans_clip'] = np.clip(df_bold_sim['corr_obs_trans'], -1.5, 1.5)

#%%
bold_pwr_dict = {"method": [], "power": [], "participants": []}
for method in ["lsa", "lss"]:
    # from 5 participants to 40
    for participants in range(5, 41):
        bold_test_df, bold_pwr = test_power(
            df_bold_sim,
            estimation_method=method,
            iti_mean=18.84,
            n_trials=50,
            signal_magnitude=1.7502511308736108,
            correlation_tgt1=0.0,
            correlation_tgt2=0.1,
            sample_size=participants)
        bold_pwr_dict['method'].append(method)
        bold_pwr_dict['participants'].append(participants)
        bold_pwr_dict['power'].append(bold_pwr)

bold_pwr_df = pd.DataFrame.from_dict(bold_pwr_dict)
bold_pwr_df.head()

#%%
pp = sns.lineplot(
    x='participants',
    y='power',
    hue='method',
    legend="brief",
    hue_order=["lss", "lsa"],
    data=bold_pwr_df)

lgnd = pp.legend()
lgnd.get_texts()[0].set_text('')
for txt in lgnd.get_texts()[1:]:
    txt.set_fontsize(15)
lgnd.set_title("Estimation Method", prop={'size': 15, 'weight': 'heavy'})
pp.set_xlabel("# of Participants",
              fontdict={'fontsize': 15, 'fontweight': 'heavy'})
pp.set_ylabel("Power",
              fontdict={'fontsize': 15, 'fontweight': 'heavy'})
pp.axhline(0.8, linestyle='--')
plt.tight_layout()
pp.figure.savefig('outputs/power_plot.svg')
pp.figure.savefig('outputs/power_plot.png', dpi=400)
#%%
