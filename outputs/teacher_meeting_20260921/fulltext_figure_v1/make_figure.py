from pathlib import Path
import xml.etree.ElementTree as E
O=Path('outputs/teacher_meeting_20260921/fulltext_figure_v1');O.mkdir(exist_ok=True)
r=E.Element('mxfile');d=E.SubElement(r,'diagram',id='fulltext-method',name='Company-focused full text');m=E.SubElement(d,'mxGraphModel',page='1',pageWidth='1800',pageHeight='830',grid='1',gridSize='8');root=E.SubElement(m,'root');E.SubElement(root,'mxCell',id='0');E.SubElement(root,'mxCell',id='1',parent='0')
navy='#17324D';teal='#006E73';blue='#EDF3F9';pale='#E7F4F1';muted='#536676'
def node(i,t,x,y,w,h,accent=False,text=False,fs=30,bold=False,align='center'):
 st=f'rounded=1;whiteSpace=wrap;html=0;fillColor={pale if accent else blue};strokeColor={"#88B7B1" if accent else "#B7C9DB"};strokeWidth=2;fontFamily=Helvetica Neue;fontSize={fs};fontStyle={1 if bold else 0};fontColor={teal if accent else navy};align={align};verticalAlign=middle;'
 if text:st='text;'+st+'fillColor=none;strokeColor=none;'
 c=E.SubElement(root,'mxCell',id=i,value=t,style=st,vertex='1',parent='1');E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
def edge(i,s,t,pts,dash=False):
 c=E.SubElement(root,'mxCell',id=i,source=s,target=t,edge='1',parent='1',style=f'edgeStyle=none;html=0;endArrow=block;endFill=1;strokeColor={muted};strokeWidth=2;'+('dashed=1;' if dash else ''))
 g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});E.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),attrib={'as':'sourcePoint'});a=E.SubElement(g,'Array',attrib={'as':'points'})
 for x,y in pts[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
 E.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),attrib={'as':'targetPoint'})
node('input','News available\nTitles + bodies\nSource + time',24,145,244,184,fs=27)
node('select','1  Select passages\nCompany + context\nKeep negation,\nnumbers and IDs',344,145,292,184,True,fs=27)
node('group','2  Group repeats\nSimilar text + cues\nKeep extra details\nSplit conflicts',712,145,304,184,True,fs=27)
node('encode','3  Encode + pool\nFrozen FinBERT\nMean within groups,\nthen across groups\nTitle summary',1092,145,308,184,fs=27)
node('predict','4  Predict\nJoin all features\nRegularized LR',1476,145,300,184,fs=27)
edge('e1','input','select',[(268,237),(344,237)]);edge('e2','select','group',[(636,237),(712,237)]);edge('e3','group','encode',[(1016,237),(1092,237)]);edge('e4','encode','predict',[(1400,237),(1476,237)])
edge('titlepath','input','encode',[(146,145),(146,55),(1246,55),(1246,145)])
node('title_label','Titles remain a separate input',430,8,540,42,text=True,fs=26)
node('meta','Reporting + coverage features\nCounts, sources, ages and body coverage',344,410,672,104,fs=28)
edge('metadata','input','meta',[(146,329),(146,462),(344,462)])
edge('selectedcoverage','select','meta',[(490,329),(490,410)])
edge('groupcounts','group','meta',[(864,329),(864,410)])
edge('metapred','meta','predict',[(1016,462),(1440,462),(1440,303),(1476,303)])
node('prices','Historical prices\nPast completed bars',1092,504,308,96,fs=27)
edge('pricepred','prices','predict',[(1400,552),(1810,552),(1810,277),(1776,277)])
# Route price inside final canvas, right gutter reserved by narrowing prediction.
for c in root.findall('mxCell'):
 if c.get('id')=='predict': c.find('mxGeometry').set('width','276')
 if c.get('id')=='pricepred':
  g=c.find('mxGeometry');g.find("mxPoint[@as='targetPoint']").set('x','1752')
  for p in g.findall('Array/mxPoint'):p.set('x','1784')
node('out','4-hour\nup probability',1500,370,228,92,fs=29,bold=True)
edge('output','predict','out',[(1614,329),(1614,370)])
node('legend','Teal = proposed text processing     Blue = standard inputs / models',24,770,1390,40,text=True,fs=25,align='left')
node('sample_title','ILLUSTRATION: preserve the extra detail',24,548,1000,58,text=True,fs=34,bold=True,align='left')
node('ex_input','Report A: target price cut to 180\nReport B: same cut + new guidance',24,610,575,115,fs=29)
node('ex_output','One target-price group\nOne additional guidance block',712,610,605,115,True,fs=29)
edge('example','ex_input','ex_output',[(599,667),(712,667)])
node('fallback','No reliable body: keep title\nNo news: price-only fallback',1370,666,405,74,text=True,fs=22,align='left')
E.indent(r);E.ElementTree(r).write(O/'fulltext_method.drawio',encoding='utf-8',xml_declaration=True)
(O/'brief.md').write_text('''# Slide 6: company-focused full-text extension\nAudience: CSE573 faculty. Proposed architecture, not a completed or scored experiment.\nContent authority: user discussion of company-paragraph selection, title preservation, passage-level grouping, frozen FinBERT, metadata and price inputs. Existing slide is style only.\nMust show target-company selection, context/negation/numeric preservation, repeated-block grouping with unmatched details retained, separate titles, metadata/coverage, prices, classifier, 4h output, missing-body/news fallback.\nNo invented feature dimensions or results; no claim of verified event identity or market-first disclosure. Fixed pooling first; no attention or LLM claimed.\nThe example is illustrative, not a historical case.\nInput is already cutoff-qualified news. Unknown or unreliable body selection falls back to titles. Feature scaling/compression/classifier fit on past training only. Sentence IDs retained for inspection.\nSemantic traceability: input->selection->grouping->encoding/group pooling->prediction; titles bypass body processing into encoding; metadata derived from raw reports plus selection/group records; prices go directly to classifier.\n''')
(O/'visual-spec.md').write_text('''# Visual contract\nInherited deck: white, Helvetica Neue, navy #17324D. Proposed modules use teal #006E73 with pale #E7F4F1; standard modules #EDF3F9. 1800x830 canvas. Main labels27-30px, main slide title34px at1280x720. Editable rectangles and explicit arrows. Single left-to-right dominant path; bottom teaching example. Colors denote role, not performance. Native vector translation into PPTX. PDF exports use embedded Arial for portable glyphs.\n''')
(O/'asset-ledger.md').write_text('No external assets. All computation and illustrative examples are native editable vector primitives. No generated raster assets.\n')
(O/'layout-grid.md').write_text('''# Layout\nFive aligned stages y145..329; x24,344,712,1092,1476. Title bypass y55. Reporting lane y410. Price lane y548. Example y610. Solid arrows are data. No edges may cross unrelated boxes. Group records and selection coverage must also feed reporting metadata.\n''')
