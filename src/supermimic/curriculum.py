"""Promote forward speed only after sustained real-simulator competence."""
class SpeedCurriculum:
    def __init__(self, maximum=5., initial=1.):
        self.maximum=maximum;self.speed=min(initial,maximum);self.streak=0
        self.totals=dict(steps=0,upright=0,moving=0,tracking=0,episodes=0,length=0)

    def add(self, **values):
        for k,v in values.items():self.totals[k]+=float(v)

    def update(self):
        t=self.totals
        up=t['upright']/max(t['steps'],1)
        tracking=t['tracking']/max(t['moving'],1)
        length=t['length']/max(t['episodes'],1)
        enough=t['episodes']>=256 and t['moving']>=4096
        qualified=enough and up>=.9 and tracking>=.75 and length>=15.
        self.streak=self.streak+1 if qualified else 0
        if self.streak>=2:
            self.speed=min(self.maximum,self.speed+.5);self.streak=0
        result=dict(curriculum_max_speed=self.speed,curriculum_upright_fraction=up,
            curriculum_tracking_fraction=tracking,curriculum_mean_episode_seconds=length,
            curriculum_qualified_streak=self.streak)
        self.totals={k:0 for k in t}
        return result

    def state_dict(self):return dict(speed=self.speed,streak=self.streak,totals=self.totals.copy())
    def load_state_dict(self,state):
        self.speed=state['speed'];self.streak=state['streak'];self.totals=state['totals'].copy()
