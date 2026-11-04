"""Three matched 40k policies: fixed recovery, walking/running and turning clips."""
import argparse
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
root = p.parse_args().directory.resolve()
font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
models = [
 ('AMP-shared', 'AMP｜分层reset＋共享单判别器＋恢复均衡采样\n复杂地形高程critic｜40k'),
 ('SuperMimic-original-height', 'SuperMimic｜共享单判别器＋复杂地形\n原高度奖励｜40k'),
 ('SuperMimic-half-height', 'SuperMimic｜共享单判别器＋复杂地形\n半强度高度奖励｜40k'),
]
cases = [('supine','仰卧起身'),('prone','俯卧起身'),('left_side_down','左侧卧起身'),('right_side_down','右侧卧起身'),('walk_run','站立→0.5→1→2→3 m/s，每档5秒'),('turn','原地转向：0→1→−1→1→−1 rad/s，每档5秒')]
records = [json.loads((root / name / 'recording.json').read_text()) for name,_ in models]
assert len({r['initial_state_hash'] for r in records}) == 1
assert len({r['bank_sha256'] for r in records}) == 1
for i,(_,title) in enumerate(models):
 (root / f'title_{i}.txt').write_text(title)
outputs = []
for case,desc in cases:
 heading = root / f'heading_{case}.txt'
 heading.write_text(f'{desc}｜相同初始姿态，平地，原速播放，无reset')
 cmd = ['ffmpeg','-v','error','-y']
 for name,_ in models:
  cmd += ['-i',str(root / name / f'{case}.mp4')]
 filters = [f'[{i}:v]pad=640:540:0:76:color=0x14181e,drawtext=fontfile={font}:textfile={root / f"title_{i}.txt"}:fontsize=20:fontcolor=white:x=8:y=6:line_spacing=6[t{i}]' for i in range(3)]
 filters.append(f'[t0][t1][t2]hstack=inputs=3,pad=1920:592:0:52:color=0x14181e,drawtext=fontfile={font}:textfile={heading}:fontsize=25:fontcolor=white:x=15:y=10[v]')
 dest = root / f'comparison_{case}.mp4'
 cmd += ['-filter_complex',';'.join(filters),'-map','[v]','-c:v','libx264','-crf','20','-threads','2','-movflags','+faststart',str(dest)]
 subprocess.run(cmd,check=True)
 outputs.append(str(dest))
concat = root / 'concat_recovery.txt'
concat.write_text(''.join(f"file 'comparison_{case}.mp4'\n" for case,_ in cases[:4]))
recovery = root / 'comparison_recovery.mp4'
subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(recovery)],check=True)
(root / 'comparison_manifest.json').write_text(json.dumps({'layout':dict(zip(['left','middle','right'],[v[1] for v in models])),'recordings':records,'outputs':outputs+[str(recovery)]},ensure_ascii=False,indent=2)+'\n')
