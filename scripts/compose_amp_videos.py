from pathlib import Path
import imageio.v2 as iio
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import argparse
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);p=parser.parse_args().directory
names=['baseline','dual','strat_single','strat_dual','guidance']
titles=['Original reset | Single D','Original reset | Dual D','Stratified reset | Single D','Stratified reset | Dual D','Stratified | Dual D + recovery reward']
scenes=['supine','prone','left_side','right_side','natural_prone','walking']
f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
readers=[iio.get_reader(p/(n+'.mp4')) for n in names]
writers=[iio.get_writer(p/('compare_'+s+'.mp4'),fps=12.5,codec='libx264',quality=7,ffmpeg_params=['-threads','2','-movflags','+faststart']) for s in scenes]
for t,frames in enumerate(zip(*readers)):
 for j,s in enumerate(scenes):
  canvas=Image.new('RGB',(1920,832),(20,24,30));d=ImageDraw.Draw(canvas)
  for k,frame in enumerate(frames):
   x=(k%3)*640;y=(k//3)*416;sx=(j%3)*640;sy=(j//3)*416
   crop=Image.fromarray(frame).crop((sx,sy+32,sx+640,sy+416)).resize((640,384))
   canvas.paste(crop,(x,y+32));d.text((x+8,y+3),titles[k],font=f,fill='white')
  d.text((1295,480),s.replace('_',' ').upper(),font=f,fill='white')
  d.text((1295,530),'60k checkpoints | %.2f s'%(t*.08),font=f,fill='white')
  d.text((1295,580),'Same initial pose; no resets',font=f,fill='white')
  d.text((1295,630),'20 s rollout; normal playback speed',font=f,fill='white')
  writers[j].append_data(np.asarray(canvas))
  if t in [0,25,75,200] and j==1:canvas.save(p/('prone_frame_'+str(t)+'.jpg'))
for w in writers:w.close()
for r in readers:r.close()
print('Completed six comparisons',flush=True)
