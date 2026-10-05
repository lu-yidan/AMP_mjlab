"""Frozen recovery protocol: begin standing by 20 s and confirm for 3 s."""
import numpy as np

PROTOCOL = dict(name='SR3_recovery20_confirm3_v1', recovery_deadline_s=20.,
    confirmation_s=3., observation_s=23., head_height_m=1.15,
    uprightness=.90, minimum_each_foot_load_n=20.,
    escape_required=False, quiet_motion_required=False,
    historical_plate_invalid_exclusion=False,
    deadline_convention='standing interval begins at or before t=20; ends by t=23',
    denominator='all scheduled initial trials; no retries or historical invalid exclusion')


def score(good, alive, dt=.02):
    """Rows are control intervals [k*dt,(k+1)*dt], judged at interval end.

    Thus intervals 1000..1149 qualify as the complete [20,23] s hold.
    Integer counters avoid float accumulation changing the boundary.
    """
    good, alive = np.asarray(good, bool), np.asarray(alive, bool)
    required = round(3/dt); deadline = round(20/dt); horizon = round(23/dt)
    assert good.shape == alive.shape and good.shape[0] >= horizon
    n = good.shape[1]; streak = np.zeros(n, np.int32); best = streak.copy()
    first = np.full(n, -1, np.int32); ever_dead = np.zeros(n, bool)
    for k in range(horizon):
        ever_dead |= ~alive[k]
        streak = np.where(good[k] & ~ever_dead, streak+1, 0)
        best = np.maximum(best, streak)
        eligible = (streak >= required) & (k+1-streak <= deadline)
        first[(first < 0) & eligible] = k+1
    return dict(success=first >= 0, first_confirmation_step=first,
                longest_standing_s=best*dt, terminated=ever_dead)


def summarize(result, scene, direction):
    rows = {}
    masks = {'all':np.ones(len(scene),bool), 'plates':scene>0,
             'flat':scene==0, 'guided_plate':scene==1, 'free_plate':scene==2}
    for name, mask in list(masks.items()):
        if name in ('flat','guided_plate','free_plate'):
            for d, label in enumerate(('supine','prone','left','right')):
                masks[name+'/'+label] = mask & (direction==d)
    for name, mask in masks.items():
        n = int(mask.sum()); k = int(result['success'][mask].sum())
        if not n: continue
        p = k/n; z = 1.959963984540054
        center=(p+z*z/(2*n))/(1+z*z/n)
        half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        rows[name]=dict(n=n, success_n=k, sr3=100*p,
            wilson95_pct=[100*(center-half),100*(center+half)],
            terminated_n=int(result['terminated'][mask].sum()))
    return rows


def self_test():
    g=np.zeros((1150,6),bool); a=np.ones_like(g)
    g[1000:,0]=True; g[1001:,1]=True; g[0:150,2]=True
    g[:149,3]=True; g[150:299,3]=True
    g[:150,4]=True; a[50:,4]=False
    g[:150,5]=True; a[200:,5]=False
    assert score(g,a)['success'].tolist()==[True,False,True,False,False,True]

if __name__=='__main__':
    self_test(); print('SR3 boundary and reset tests passed')
