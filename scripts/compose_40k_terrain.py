"""Compose three matched40k terrain recordings with explicit Chinese experiment names."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--speeds',type=int,nargs='+',default=[1,2]);a=p.parse_args();root=a.directory.resolve()
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
models=[('AMP-shared-terrain','AMP｜分层reset＋共享单判别器＋恢复均衡采样\n复杂地形高程critic｜40k'),('SuperMimic-original-height-terrain','SuperMimic｜共享单判别器＋复杂地形\n原高度奖励｜40k'),('SuperMimic-half-height-terrain','SuperMimic｜共享单判别器＋复杂地形\n半强度高度奖励｜40k')]
terrains=[('slope','正金字塔坡面'),('slope_inverted','倒金字塔坡面'),('stairs','正金字塔台阶'),('stairs_inverted','倒金字塔台阶')]
manifest={'layout':{position:title for position,(_,title) in zip(['left','middle','right'],models)},'speed':1,'outputs':[]}
for i,(_,title) in enumerate(models):(root/f'title_{i}.txt').write_text(title)
for speed in a.speeds:
 records=[json.loads((root/(model+f'-vx{speed}')/'terrain.json').read_text()) for model,_ in models]
 assert len({r['initial_state_hash'] for r in records})==1
 clips=[]
 for terrain,chinese in terrains:
  heading=root/f'heading_{terrain}_{speed}.txt';heading.write_text(f'{chinese}（坡15° / 台阶10cm）｜目标 {speed} m/s｜每段先静止2秒，再前进10秒｜原速，无reset')
  cmd=['ffmpeg','-v','error','-y']
  for model,_ in models:cmd+=['-i',str(root/(model+f'-vx{speed}')/(terrain+'.mp4'))]
  filters=[f"[{i}:v]pad=640:520:0:72:color=0x14181e,drawtext=fontfile={font}:textfile={root/f'title_{i}.txt'}:fontsize=21:fontcolor=white:x=10:y=6:line_spacing=6[t{i}]" for i in range(3)]
  filters.append(f"[t0][t1][t2]hstack=inputs=3,pad=1920:576:0:56:color=0x14181e,drawtext=fontfile={font}:textfile={heading}:fontsize=24:fontcolor=white:x=15:y=12[v]")
  dest=root/f'comparison_{terrain}_{speed}mps.mp4';cmd+=['-filter_complex',';'.join(filters),'-map','[v]','-c:v','libx264','-crf','20','-threads','2','-movflags','+faststart',str(dest)];subprocess.run(cmd,check=True);clips.append(dest)
 concat=root/f'concat_{speed}.txt';concat.write_text(''.join(f"file '{f.name}'\n" for f in clips));dest=root/f'terrain_three_40k_{speed}mps.mp4'
 subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(dest)],check=True)
 manifest['outputs'].append(dict(command_speed=speed,path=str(dest),terrain_order=[v[1] for v in terrains],recordings=records))
(root/('terrain_comparison_manifest_'+('-'.join(map(str,a.speeds)))+'.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
