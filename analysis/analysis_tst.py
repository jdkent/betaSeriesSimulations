
#%%
import seaborn as sns
import pandas as pd

#%%
df = pd.read_csv('data/test.tsv', sep='\t')

#%%
import numpy as np
df['corr_obs_trans'] = np.arctanh(df['correlation_observed'])
df['corr_target_trans'] = np.arctanh(df['correlation_target'])

#%%
# Draw a categorical scatterplot to show each observation
g = sns.stripplot(x="correlation_target", y="corr_obs_trans", hue="estimation_method",
                  dodge=True, jitter=True, zorder=1, data=df)
g.axes.legend(loc='lower right')




#%%
import nipype.pipeline.engine as pe

def A():
    subject = ""
    return subject

def B(subject):
    out_file = ""
    return out_file




#%%
a = pe.Node(interface=A(), name="a")
b = pe.Node(interface=B(), name="b")
b.iterables = ("in_file", images)
c = pe.Node(interface=C(), name="c")
d = pe.JoinNode(interface=D(), joinsource="b",
                joinfield="in_files", name="d")

my_workflow = pe.Workflow(name="my_workflow")
my_workflow.connect([(a,b,[('subject','subject')]),
                     (b,c,[('out_file','in_file')])
                     (c,d,[('out_file','in_files')])
                     ])

#%%
