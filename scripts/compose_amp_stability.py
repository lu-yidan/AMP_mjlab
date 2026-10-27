"""Compose synchronized four-policy panels; validate identical initial states."""
import argparse,json,subprocess
from pathlib import Path
import imageio_ffmpeg
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
root=a.directory;codes=['AMP-A','AMP-B','AMP-C','AMP-D'];records=[json.loads((root/c/'recording.json').read_text())for c in codes]
assert len({r['initial_state_hash']for r in records})==1,'Initial states differ'
assert len({r['bank_sha256']for r in records})==1,'Banks differ'
out=root/'comparison';out.mkdir(exist_ok=True)
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
for scene in ('walk_run','turn','supine','prone','left_side_down','right_side_down'):
 cmd=[ffmpeg,'-y','-loglevel','error']
 for code in codes:cmd+=['-i',str(root/code/(scene+'.mp4'))]
 cmd+=['-filter_complex_threads','1','-filter_complex','[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]','-map','[v]','-c:v','libx264','-preset','fast','-crf','24','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(out/(scene+'.mp4'))]
 subprocess.run(cmd,check=True)
 for seconds in (3,8):
  subprocess.run([ffmpeg,'-y','-loglevel','error','-ss',str(seconds),'-i',str(out/(scene+'.mp4')),'-frames:v','1',str(out/(scene+f'_{seconds}s.jpg'))],check=True)
 print('COMPOSED '+scene,flush=True)
order=['supine','prone','left_side_down','right_side_down']
(out/'recovery_concat.txt').write_text(''.join("file '"+name+".mp4'\n" for name in order))
subprocess.run([ffmpeg,'-y','-loglevel','error','-f','concat','-safe','1','-i',str(out/'recovery_concat.txt'),'-c','copy','-movflags','+faststart',str(out/'recovery_all.mp4')],check=True)
(out/'manifest.json').write_text(json.dumps(dict(layout={'top_left':'AMP-A','top_right':'AMP-B','bottom_left':'AMP-C','bottom_right':'AMP-D'},recordings=records),indent=2))
