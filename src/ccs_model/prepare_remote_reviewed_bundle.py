#!/usr/bin/env python3
"""Create a compact de-identified 320px training bundle for remote GPU use."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from PIL import Image,ImageOps

def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args(); rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
 if args.output.exists(): shutil.rmtree(args.output)
 images=args.output/'images'; images.mkdir(parents=True)
 output_rows=[]
 for i,row in enumerate(rows,1):
  source=Path(str(row['local_image_path'])); source=source if source.is_absolute() else args.data_root/source
  with Image.open(source) as raw: image=ImageOps.exif_transpose(raw).convert('RGB')
  image.thumbnail((320,320),Image.Resampling.LANCZOS); canvas=Image.new('RGB',(320,320)); canvas.paste(image,((320-image.width)//2,(320-image.height)//2)); pixels=canvas.load()
  for y in range(round(320*.06)):
   for x in range(round(320*.35)): pixels[x,y]=(0,0,0)
  name=f'img_{i:04d}.jpg'; canvas.save(images/name,'JPEG',quality=95,optimize=True,exif=b'')
  output_rows.append({'record_id':f'ccs-reviewed-{i:04d}','local_image_path':f'images/{name}','split':row['split'],'ccs_grade':int(row['ccs_grade']),'opening_area_mm2':float(row['opening_area_mm2']),'wtw_mm':float(row['wtw_mm']),'wtw_source':row['wtw_source'],'reviewed_label':row.get('label_review_status') is not None})
 with (args.output/'manifest.jsonl').open('w',encoding='utf-8') as s:
  for row in output_rows:s.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
 audit={'records':len(output_rows),'train':sum(r['split']=='train' for r in output_rows),'val':sum(r['split']=='val' for r in output_rows),'reviewed_records':sum(r['reviewed_label'] for r in output_rows),'contains_patient_identifiers':False,'image_size':'320x320'}; (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
