"""Calibration-only estimators and deterministic source-ID-preserving policies."""
import itertools
import math
from functools import lru_cache

import numpy as np

from .common import POLICIES
from .workload import area, union


def key(shape):
    return f'{shape[0]}x{shape[1]}'


def summarize_trials(values):
    a=np.asarray(values,dtype=float)
    return {'mean_ms':float(a.mean()),'median_ms':float(np.median(a)),
            'se_ms':float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else None,'n':len(a)}


def fit_models(table):
    from experiments.cost_model_validation.models import fit_linear_model, fit_piecewise_model
    from scipy.optimize import lsq_linear
    shapes=[tuple(map(int,s.split('x'))) for s in sorted(table)]
    x=np.asarray([h*w for h,w in shapes],dtype=float)
    y=np.asarray([np.mean(table[key(s)])/1000 for s in shapes])
    fit=fit_linear_model(x,y)
    if fit.coefficients['b0'] <= 0 or fit.coefficients['b1'] <= 0:
        raise ValueError('Affine fit has nonpositive overhead/slope; run a separate calibration pilot, do not tune on test')
    # Reuse repository hinge fit for initial calibration breakpoint diagnostics.
    initial=fit_piecewise_model(x,y)
    candidates=sorted(set(x))
    candidates=[b for b in candidates if (x<=b).sum()>=3 and (x>b).sum()>=3]
    best=None
    for b in candidates:
        design=np.column_stack((np.ones_like(x),np.minimum(x,b),np.maximum(0,x-b)))
        solved=lsq_linear(design,y,bounds=(1e-15,np.inf))
        loss=float(np.sum((design@solved.x-y)**2))
        if best is None or loss<best[0]:
            best=(loss,float(b),solved.x)
    if best is None:
        raise ValueError('Insufficient calibration support for a constrained piecewise fit')
    _,b,c=best
    models={'affine':{'b0':fit.coefficients['b0'],'b1':fit.coefficients['b1']},
            'piecewise':{'b0':float(c[0]),'b1':float(c[1]),'b2':float(c[2]-c[1]),'breakpoint':b,
                         'unconstrained_initial_breakpoint':initial.breakpoint}}
    return {'models':models,'lookup':{key(s):summarize_trials(table[key(s)]) for s in shapes},
            'domain':{'min_h':min(s[0] for s in shapes),'max_h':max(s[0] for s in shapes),
                      'min_w':min(s[1] for s in shapes),'max_w':max(s[1] for s in shapes)},
            'lookup_entries':len(shapes),'selection':'calibration RSS, nonnegative intercept and branch slopes'}


class Estimator:
    """Only calibration artifacts enter this deployable object; no held-out timings."""
    def __init__(self,artifact,policy,options):
        if artifact.get('role','calibration') != 'calibration':
            raise ValueError('Deployable estimator rejects held-out measurements')
        if policy not in POLICIES or policy=='oracle':
            raise ValueError('Deployable estimator cannot load oracle/test timings')
        self.artifact,self.policy,self.options=artifact,policy,options

    def estimate(self,shape,strict=False):
        name=self.policy
        if name in ('affine','piecewise'):
            c=self.artifact['models'][name]
            a=shape[0]*shape[1]
            seconds=c['b0']+c['b1']*a
            if name=='piecewise':
                seconds+=c['b2']*max(0,a-c['breakpoint'])
            return seconds*1000,0.0,True
        table=self.artifact['lookup']
        exact=table.get(key(shape))
        if exact is not None:
            return exact['mean_ms'],exact['se_ms'],True
        if strict or name=='conservative_lookup':
            return None,None,False
        nearest=min(table,key=lambda k:(sum((int(v)-s)**2 for v,s in zip(k.split('x'),shape)),k))
        entry=table[nearest]
        return entry['mean_ms'],entry['se_ms'],False

    def decision(self,a,b,merged):
        values=[self.estimate(s,strict=self.policy=='conservative_lookup') for s in (a,b,merged)]
        if any(v[0] is None for v in values):
            return {'merge':False,'gain_ms':None,'fallback':'unmeasured_shape'}
        gain=values[0][0]+values[1][0]-values[2][0]
        threshold=self.options['minimum_gain_ms']
        lcb=gain
        if self.policy=='conservative_lookup':
            if any(v[1] is None for v in values):
                return {'merge':False,'gain_ms':gain,'fallback':'missing_uncertainty'}
            lcb-=self.options['confidence_z']*sum(v[1] for v in values)
        return {'merge':lcb>threshold,'gain_ms':gain,'gain_lcb_ms':lcb,
                'fallback':None,'all_exact':all(v[2] for v in values)}


class OracleCosts:
    """Retrospective reference. Constructor requires the held-out role label."""
    def __init__(self,artifact):
        if artifact.get('role')!='heldout_reference' or artifact.get('calibration_id') is None:
            raise ValueError('Oracle must use independently measured held-out reference trials')
        self.table=artifact['table']

    def cost(self,shape):
        return float(np.mean(self.table[key(shape)]))

    def decision(self,a,b,merged):
        gain=self.cost(a)+self.cost(b)-self.cost(merged)
        return {'merge':gain>0,'gain_ms':gain,'fallback':None}


def group(roi):
    return {'ids':[roi['roi_id']],'box':list(roi['box'])}


def merged_group(a,b):
    return {'ids':sorted(a['ids']+b['ids']),'box':union(a['box'],b['box'])}


def plan(rois,policy,shape,options,estimator=None,oracle=None):
    groups=[group(r) for r in rois]
    trace=[]
    if policy=='no_merge':
        return groups,trace,'control'
    if policy=='oracle' and len(rois)<=options['exact_oracle_max_rois']:
        if oracle is None:
            raise ValueError('Oracle reference missing')
        n=len(rois)
        if n==0:
            return [],trace,'exact_additive_partition'
        @lru_cache(None)
        def subset(mask):
            chosen=[groups[i] for i in range(n) if mask&(1<<i)]
            result=chosen[0]
            for item in chosen[1:]:
                result=merged_group(result,item)
            return result
        @lru_cache(None)
        def solve(mask):
            if not mask:
                return 0.0,()
            bit=mask&-mask
            best=None
            sub=mask
            while sub:
                if sub&bit:
                    rest_cost,rest=solve(mask^sub)
                    candidate=(oracle.cost(shape(subset(sub)['box']))+rest_cost,(sub,*rest))
                    if best is None or candidate<best:
                        best=candidate
                sub=(sub-1)&mask
            return best
        cost,masks=solve((1<<n)-1)
        return [subset(m) for m in masks],[{'reference_total_ms':cost}],'exact_additive_partition'
    if policy=='geometric':
        # Faithful simple_roi_merge_v2 sequential order, including equality at gamma.
        used=set(); final=[]
        for i in range(len(groups)):
            if i in used:
                continue
            current=groups[i];used.add(i)
            changed=True
            while changed:
                changed=False
                for j in range(len(groups)):
                    if j in used:
                        continue
                    candidate=merged_group(current,groups[j])
                    ratio=area(candidate['box'])/(area(current['box'])+area(groups[j]['box']))
                    accepted=ratio<=options['geometric_area_ratio_max']
                    trace.append({'left':current['ids'],'right':groups[j]['ids'],'area_ratio':ratio,'merge':accepted})
                    if accepted:
                        current=candidate;used.add(j);changed=True
            final.append(current)
        return final,trace,'sequential_geometric'
    if policy=='oracle' and oracle is None:
        raise ValueError('Oracle reference missing')
    while len(groups)>1:
        best=None
        for i,j in itertools.combinations(range(len(groups)),2):
            merged=merged_group(groups[i],groups[j])
            shapes=[shape(g['box']) for g in (groups[i],groups[j],merged)]
            decision=(oracle if policy=='oracle' else estimator).decision(*shapes)
            trace.append({'left':groups[i]['ids'],'right':groups[j]['ids'],'shapes':shapes,**decision})
            if decision['merge'] and (best is None or decision['gain_ms']>best[0]):
                best=(decision['gain_ms'],i,j,merged)
        if best is None:
            break
        _,i,j,merged=best
        groups=[g for k,g in enumerate(groups) if k not in (i,j)]+[merged]
    ids=[v for g in groups for v in g['ids']]
    if sorted(ids)!=sorted(r['roi_id'] for r in rois):
        raise AssertionError('Partition lost/duplicated source IDs')
    return groups,trace,'greedy_measured_reference' if policy=='oracle' else 'greedy_predicted_cost'
