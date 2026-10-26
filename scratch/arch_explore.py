import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
rows=[]
for y in range(1996,2025):
    b=pd.read_parquet(f'../data/stints/box_{y}.parquet').set_index('player_id')
    w=pd.read_csv(f'../out/war_{y}.csv').set_index('player_id')[['name','poss']]
    d=b.join(w,how='inner'); d=d[d.poss>=1500]
    p=d.poss/100
    f=pd.DataFrame({'usg':(d.fga+0.44*d.fta+d.tov)/p,'three':d.fg3a/d.fga,'rim':d.rima/d.fga,
        'astd':d.fgm_astd/d.fgm,'ast':d.ast/p,'reb':d.reb/p,'stl':d.stl/p,'blk':d.blk/p,'ftr':d.fta/d.fga},index=d.index)
    f=(f-f.mean())/f.std()  # within-season: role relative to that era
    f['season']=y; f['name']=d.name
    rows.append(f)
F=pd.concat(rows).reset_index()
cols=['usg','three','rim','astd','ast','reb','stl','blk','ftr']
for k in (6,7,8,9):
    km=KMeans(k,n_init=20,random_state=0).fit(F[cols]); print(k, round(km.inertia_))
km=KMeans(8,n_init=20,random_state=0).fit(F[cols]); F['c']=km.labels_
cen=pd.DataFrame(km.cluster_centers_,columns=cols).round(2); cen['n']=F.c.value_counts().sort_index()
print(cen.to_string())
for c in range(8):
    print(c, F[(F.c==c)&(F.season==2024)].name.head(8).tolist())
