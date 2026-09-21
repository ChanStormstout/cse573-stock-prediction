from pathlib import Path
import xml.etree.ElementTree as E
O=Path('outputs/teacher_meeting_20260921/reference_fulltext_v4');O.mkdir(exist_ok=True)
r=E.Element('mxfile');d=E.SubElement(r,'diagram',id='fulltext-method',name='Company-focused full text');m=E.SubElement(d,'mxGraphModel',page='1',pageWidth='2040',pageHeight='850',grid='1',gridSize='8');root=E.SubElement(m,'root');E.SubElement(root,'mxCell',id='0');E.SubElement(root,'mxCell',id='1',parent='0')
navy='#1E314B';teal='#2E6C72';blue='#EEF3F8';pale='#E9F4F1';muted='#607181'
def node(i,t,x,y,w,h,accent=False,text=False,fs=30,bold=False,align='center'):
 st=f'rounded=1;whiteSpace=wrap;html=0;fillColor={pale if accent else blue};strokeColor={"#88B7B1" if accent else "#B7C9DB"};strokeWidth=2;fontFamily=Helvetica Neue;fontSize={fs};fontStyle={1 if bold else 0};fontColor={teal if accent else navy};align={align};verticalAlign=middle;'
 if text:st='text;'+st+'fillColor=none;strokeColor=none;'
 c=E.SubElement(root,'mxCell',id=i,value=t,style=st,vertex='1',parent='1');E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
def edge(i,s,t,pts,dash=False):
 c=E.SubElement(root,'mxCell',id=i,source=s,target=t,edge='1',parent='1',style=f'edgeStyle=none;html=0;endArrow=block;endFill=1;strokeColor={muted};strokeWidth=2;'+('dashed=1;' if dash else ''))
 g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});E.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),attrib={'as':'sourcePoint'});a=E.SubElement(g,'Array',attrib={'as':'points'})
 for x,y in pts[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
 E.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),attrib={'as':'targetPoint'})
node('source','Available news\nTitles + bodies\nSource / time',24,270,240,165,fs=28,bold=True)
node('prepare','Prepare text\nTarget passages\n+ original titles',334,270,270,165,True,fs=26,bold=True)
node('encoder','FinBERT\nFrozen encoder',674,100,270,140,fs=28,bold=True)
node('groups','Group passages\nText + fact cues\nKeep extra text',674,415,270,160,True,fs=27,bold=True)
node('pool','Body: two-level mean\nWithin, then across\nTitles: own average',1024,85,340,175,True,fs=28,bold=True)
node('report','Reporting features\nCounts / sources\nAges / coverage',1024,415,340,150,True,fs=28,bold=True)
node('price','Historical prices\nCompleted bars',1024,670,340,100,fs=32,bold=True)
node('combine','Combine\nTitle + body\nReporting\nPrices',1444,275,230,225,fs=32,bold=True)
node('model','Logistic\nregression\nTrained',1744,280,250,170,fs=34,bold=True)
node('out','4-hour up\nprobability',1744,620,250,100,fs=32,bold=True)
edge('raw','source','prepare',[(264,352),(334,352)])
edge('encode','prepare','encoder',[(469,270),(469,170),(674,170)])
edge('group','prepare','groups',[(469,435),(469,495),(674,495)])
edge('vectors','encoder','pool',[(944,170),(1024,170)])
edge('ids','groups','pool',[(944,455),(985,455),(985,235),(1024,235)],True)
edge('reportedge','groups','report',[(944,495),(1024,495)])
edge('text','pool','combine',[(1364,170),(1404,170),(1404,320),(1444,320)])
edge('meta','report','combine',[(1364,490),(1404,490),(1404,410),(1444,410)])
edge('prices','price','combine',[(1364,720),(1424,720),(1424,460),(1444,460)])
edge('predict','combine','model',[(1674,385),(1744,385)])
edge('prob','model','out',[(1869,450),(1869,620)])
node('idslabel','body group IDs',1008,303,280,40,text=True,fs=26,align='left')
node('legend','Blue: standard components\nTeal: proposed text + reporting processing\nSolid: data   Dashed: body group membership',24,640,930,120,text=True,fs=26,align='left')
node('fallback','No reliable body → keep titles.    No admitted news → use the price-only forecast.',24,804,1970,38,text=True,fs=26,align='left')
E.indent(r);E.ElementTree(r).write(O/'method.drawio',encoding='utf-8',xml_declaration=True)
