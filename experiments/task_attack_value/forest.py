"""Portable inference for OWN exported trees; no sklearn or pickle at runtime."""
import numpy as np

_CACHE={}


def predict(model,x):
    # Match the training library's float32 split-boundary convention.
    x=np.array(x,dtype=np.float32);result=np.zeros(len(x))
    if x.ndim!=2 or x.shape[1]!=model['features'] or not np.isfinite(x).all():raise ValueError('Invalid portable forest inputs')
    key=id(model)
    if key not in _CACHE:
        if len(_CACHE)>=24:_CACHE.pop(next(iter(_CACHE)))
        compiled=[tuple(np.array(t[k]) for k in ('left','right','feature','threshold','value'))+(t['max_depth'],) for t in model['trees']]
        _CACHE[key]=(model,compiled)
    for left,right,feature,threshold,value,depth in _CACHE[key][1]:
        node=np.zeros(len(x),dtype=np.int32);leaf=left<0
        for _ in range(depth+1):
            active=~leaf[node]
            if not active.any():break
            rows=np.flatnonzero(active);current=node[rows]
            node[rows]=np.where(x[rows,feature[current]]<=threshold[current],left[current],right[current])
        result+=value[node]
    return np.clip(result/max(1,len(model['trees'])),0.,1.)


def export(model):
    trees=[]
    for estimator in model.estimators_:
        tree=estimator.tree_;trees.append(dict(left=tree.children_left.tolist(),right=tree.children_right.tolist(),
            feature=tree.feature.tolist(),threshold=tree.threshold.tolist(),value=tree.value[:,0,0].tolist(),max_depth=tree.max_depth))
    return dict(trees=trees,features=int(model.n_features_in_),kind='ExtraTreesRegressor',export='Finite numeric JSON; no executable pickle')
