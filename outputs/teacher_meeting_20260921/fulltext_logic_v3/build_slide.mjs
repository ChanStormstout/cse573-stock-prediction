import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const ROOT='/Users/victor/Documents/Codex/2026-09-14/wox';
const W=path.join(ROOT,'work/presentation_meeting/fulltext_logic_v3');
const O=path.join(ROOT,'outputs/teacher_meeting_20260921/fulltext_logic_v3');
const SKILL='/Users/victor/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11814/skills/presentations';
const p=Presentation.create({slideSize:{width:1280,height:720}});const s=p.slides.add();s.background.fill='#FFFFFF';
const C={ink:'#17324D',muted:'#627182',blue:'#37699C',pale:'#EDF3FA',teal:'#007C78',mint:'#E4F3EF',line:'#CCD6DE'};
function box(x,y,w,h,fill,stroke='none',r=0){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:stroke==='none'?0:1.3},borderRadius:r});}
function text(t,x,y,w,h,size=22,color=C.ink,bold=false,align='left'){const a=box(x,y,w,h,'none');a.text=t;a.text.style={typeface:'Helvetica Neue',fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none'};return a;}
function line(pts,col=C.line,width=1.5,arrow=false){let xs=pts.map(v=>v[0]),ys=pts.map(v=>v[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(1,Math.max(...xs)-x),h=Math.max(1,Math.max(...ys)-y);s.shapes.add({geometry:'custom',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:col,width},customPaths:[{width:w,height:h,commands:pts.map((v,i)=>({[i?'lineTo':'moveTo']:{x:v[0]-x,y:v[1]-y}}))}]});if(arrow){const [ex,ey]=pts.at(-1),[px,py]=pts.at(-2),angle=Math.atan2(ey-py,ex-px);const tri=[[ex,ey],[ex-10*Math.cos(angle)+5*Math.sin(angle),ey-10*Math.sin(angle)-5*Math.cos(angle)],[ex-10*Math.cos(angle)-5*Math.sin(angle),ey-10*Math.sin(angle)+5*Math.cos(angle)]];const tx=Math.min(...tri.map(v=>v[0])),ty=Math.min(...tri.map(v=>v[1])),tw=Math.max(1,Math.max(...tri.map(v=>v[0]))-tx),th=Math.max(1,Math.max(...tri.map(v=>v[1]))-ty);s.shapes.add({geometry:'custom',position:{left:tx,top:ty,width:tw,height:th},fill:col,line:{fill:col,width:.1},customPaths:[{width:tw,height:th,commands:[...tri.map((v,i)=>({[i?'lineTo':'moveTo']:{x:v[0]-tx,y:v[1]-ty}})),{close:{}}]}]});}}

async function icon(n,x,y,size,col=C.ink){let svg=(await fs.readFile(path.join(O,'assets',n+'.svg'),'utf8')).replaceAll('currentColor',col);s.images.add({blob:Buffer.from(svg),contentType:'image/svg+xml',alt:n+' icon, Lucide ISC license',fit:'contain',position:{left:x,top:y,width:size,height:size}});}
text('Full-text extension of our news–price model',48,28,1184,60,36,C.ink,true);
text('Proposed architecture    •    FinBERT: a financial-language encoder',50,97,1180,29,18,C.muted);
text('INPUTS',50,146,165,29,16,C.muted,true);
text('TEXT & PRICE REPRESENTATIONS',235,146,560,29,16,C.muted,true);
text('JOIN',824,146,135,29,16,C.muted,true,'center');
// Independent input lanes; teal marks only the new body route.
box(38,187,159,76,C.pale,'none',5);text('News titles',48,203,139,44,24,C.ink,true,'center');
box(38,298,159,76,C.mint,'none',5);text('Full bodies',48,314,139,44,24,C.teal,true,'center');
box(38,410,159,76,C.pale,'none',5);text('Report\nmetadata',48,416,139,62,22,C.ink,false,'center');
box(38,522,159,76,C.pale,'none',5);text('Past prices',48,538,139,44,24,C.ink,true,'center');
// Existing title route.
box(434,187,166,76,C.pale,C.line,5);text('FinBERT\n+ average',443,191,148,67,22,C.ink,false,'center');
line([[197,225],[434,225]],C.muted,1.6,true);line([[600,225],[821,225]],C.muted,1.6,true);
// New body route, left to right; preserve distinct operations.
box(218,280,576,113,'#F2F8F6','none',5);
box(232,298,174,76,C.mint,'#9CC6BD',5);text('Company\npassages',240,303,158,65,23,C.teal,true,'center');
box(434,298,136,76,C.mint,'#9CC6BD',5);text('FinBERT',441,315,122,43,22,C.teal,true,'center');
box(600,298,176,76,C.mint,'#9CC6BD',5);text('Group repeats\n+ average',608,303,160,65,20,C.teal,true,'center');
line([[197,336],[232,336]],C.teal,2,true);line([[406,336],[434,336]],C.teal,2,true);line([[570,336],[600,336]],C.teal,2,true);line([[776,336],[821,336]],C.teal,2,true);
// Reporting features receive group counts as well as raw metadata.
box(434,410,342,76,C.pale,C.line,5);text('Counts • sources • ages\nText coverage',442,416,326,62,21,C.ink,false,'center');
line([[197,448],[434,448]],C.muted,1.6,true);line([[688,374],[688,410]],C.muted,1.4,true);line([[319,374],[319,448]],C.muted,1.4,true);
line([[776,448],[821,448]],C.muted,1.6,true);
box(434,522,342,76,C.pale,C.line,5);text('Historical price features',441,538,328,44,23,C.ink,false,'center');
line([[197,560],[434,560]],C.muted,1.6,true);line([[776,560],[821,560]],C.muted,1.6,true);
// A single feature stack and a single prediction path.
for(const [t,y,col,ink] of [['Titles',187,C.pale,C.ink],['Body',298,C.mint,C.teal],['Reports',410,C.pale,C.ink],['Prices',522,C.pale,C.ink]]){box(821,y,130,76,col,'none');text(t,826,y+17,120,43,23,ink,t==='Body','center');}
line([[812,187],[803,187],[803,598],[812,598]],C.line,1.2);
line([[960,187],[969,187],[969,598],[960,598]],C.muted,1.5);
line([[969,392],[999,392]],C.muted,2,true);
box(999,345,234,95,C.ink,'none',5);text('Logistic\nregression',1010,353,212,77,27,'#FFFFFF',true,'center');
line([[1116,440],[1116,481]],C.muted,1.7,true);
text('4-hour direction',994,487,244,40,25,C.ink,true,'center');text('Up / Down',994,530,244,35,23,C.muted,false,'center');
// Small illustrative inset explains grouping; it is not an extra pipeline branch.
line([[50,625],[1230,625]],C.line,1);
text('EXAMPLE',50,653,94,30,15,C.muted,true);
box(157,643,143,47,C.pale);text('A: cut to $180',159,647,139,38,18,C.blue,false,'center');
box(310,643,143,47,C.pale);text('B: cut to $180',312,647,139,38,18,C.blue,false,'center');
line([[465,667],[494,667]],C.muted,1.6,true);text('One shared group',506,647,207,39,20,C.blue,true);
box(763,643,190,47,C.mint);text('B: new guidance',767,647,182,38,18,C.teal,false,'center');
line([[966,667],[995,667]],C.muted,1.6,true);text('Separate group',1007,647,223,39,20,C.teal,true);
s.speakerNotes.textFrame.setText('Proposed full-text extension, not a measured result. Follow the teal body route: select company passages with context, encode with frozen FinBERT, then average within near-duplicate groups and across groups. Extra details stay separate. Titles keep their own representation. Join text, reporting features and completed historical prices in logistic regression. Missing body retains titles; no news uses price-only fallback. Target-price example is illustrative.');
await (await PresentationFile.exportPptx(p)).save(path.join(W,'candidate.pptx'));
const {finalizePresentation}=await import(path.join(SKILL,'container_tools/artifact_tool_utils.mjs'));
console.log(await finalizePresentation({workspaceDir:ROOT,candidatePath:path.join(W,'candidate.pptx'),finalPath:path.join(O,'Slide6_FullText_Architecture_Final.pptx'),pythonExecutable:'/Users/victor/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],fontPolicy:{basis:'design',families:['Helvetica Neue']},explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:path.join(W,'validation_final_v2.json')}));
