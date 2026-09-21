from pathlib import Path
import xml.etree.ElementTree as ET
import json,html,math
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("ArialExport","/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont("ArialExportBold","/System/Library/Fonts/Supplemental/Arial Bold.ttf"))
O=Path('outputs/teacher_meeting_20260921/reference_fulltext_v4')
for name in ['method']:
 root=ET.parse(O/(name+'.drawio')).getroot();model=root.find('.//mxGraphModel');w,h=float(model.get('pageWidth')),float(model.get('pageHeight'));nodes=[];edges=[]
 for c in root.findall('.//mxCell'):
  g=c.find('mxGeometry')
  if g is None:continue
  st=dict(x.split('=',1) if '=' in x else (x,'1') for x in c.get('style','').split(';') if x)
  if c.get('vertex')=='1':
   nodes.append(dict(id=c.get('id'),text=c.get('value'),x=float(g.get('x')),y=float(g.get('y')),w=float(g.get('width')),h=float(g.get('height')),font=float(st.get('fontSize',32)),bold=st.get('fontStyle')=='1',fill=st.get('fillColor'),stroke=st.get('strokeColor'),color=st.get('fontColor'),align=st.get('align'),textOnly='text' in st))
  elif c.get('edge')=='1':
   pts=[g.find("mxPoint[@as='sourcePoint']"),*g.findall('Array/mxPoint'),g.find("mxPoint[@as='targetPoint']")];edges.append(dict(id=c.get('id'),source=c.get('source'),target=c.get('target'),pts=[[float(p.get('x')),float(p.get('y'))] for p in pts],dash=st.get('dashed')=='1'))
 data=dict(w=w,h=h,nodes=nodes,edges=edges);(O/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
 svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/>']
 pdf=Canvas(str(O/(name+'.pdf')),pagesize=(w,h));pdf.setTitle(name);pdf.setFillColor(HexColor('#ffffff'));pdf.rect(0,0,w,h,stroke=0,fill=1)
 for e in edges:
  pts=e['pts'];pstr=' '.join(f'{x},{y}' for x,y in pts);dash=' stroke-dasharray="8 6"' if e['dash'] else ''
  svg.append(f'<polyline points="{pstr}" fill="none" stroke="#536676" stroke-width="2.5"{dash}/>')
  pdf.setStrokeColor(HexColor('#536676'));pdf.setLineWidth(2.5);pdf.setDash([8,6] if e['dash'] else []);p=pdf.beginPath();p.moveTo(pts[0][0],h-pts[0][1])
  for x,y in pts[1:]:p.lineTo(x,h-y)
  pdf.drawPath(p);pdf.setDash([])
  x,y=pts[-1];px,py=pts[-2];theta=math.atan2(y-py,x-px);end=[[x,y],[x-13*math.cos(theta)+6*math.sin(theta),y-13*math.sin(theta)-6*math.cos(theta)],[x-13*math.cos(theta)-6*math.sin(theta),y-13*math.sin(theta)+6*math.cos(theta)]]
  svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in end)+'" fill="#536676"/>');pdf.setFillColor(HexColor('#536676'));p=pdf.beginPath();p.moveTo(end[0][0],h-end[0][1]);p.lineTo(end[1][0],h-end[1][1]);p.lineTo(end[2][0],h-end[2][1]);p.close();pdf.drawPath(p,stroke=0,fill=1)
 for n in nodes:
  if not n['textOnly']:
   svg.append(f'<rect x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{n["h"]}" rx="8" fill="{n["fill"]}" stroke="{n["stroke"]}" stroke-width="2"/>');pdf.setFillColor(HexColor(n['fill']));pdf.setStrokeColor(HexColor(n['stroke']));pdf.roundRect(n['x'],h-n['y']-n['h'],n['w'],n['h'],8,stroke=1,fill=1)
  lines=n['text'].split('\n');lineh=n['font']*1.22;y=n['y']+(n['h']-lineh*len(lines))/2+n['font'];x=n['x']+ (0 if n['align']=='left' else n['w']/2);anchor='start' if n['align']=='left' else 'middle';weight='700' if n['bold'] else '400'
  for line in lines:
   svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Helvetica Neue,Arial,sans-serif" font-size="{n["font"]}" font-weight="{weight}" fill="{n["color"]}">{html.escape(line)}</text>');pdf.setFillColor(HexColor(n['color']));pdf.setFont('ArialExportBold' if n['bold'] else 'ArialExport',n['font']);getattr(pdf,'drawString' if n['align']=='left' else 'drawCentredString')(x,h-y,line);y+=lineh
 svg.append('</svg>');(O/(name+'.svg')).write_text(''.join(svg));pdf.save()
