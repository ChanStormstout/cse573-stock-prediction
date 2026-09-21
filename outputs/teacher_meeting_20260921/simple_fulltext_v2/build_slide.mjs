import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const ROOT='/Users/victor/Documents/Codex/2026-09-14/wox';
const W=path.join(ROOT,'work/presentation_meeting/simple_fulltext_v2');
const O=path.join(ROOT,'outputs/teacher_meeting_20260921/simple_fulltext_v2');
const SKILL='/Users/victor/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11814/skills/presentations';
const p=Presentation.create({slideSize:{width:1280,height:720}});const s=p.slides.add();s.background.fill='#FFFFFF';
const C={ink:'#17324D',muted:'#627182',blue:'#37699C',pale:'#EDF3FA',teal:'#007C78',mint:'#E4F3EF',line:'#CCD6DE'};
function box(x,y,w,h,fill,stroke='none',r=0){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:stroke==='none'?0:1.3},borderRadius:r});}
function text(t,x,y,w,h,size=22,color=C.ink,bold=false,align='left'){const a=box(x,y,w,h,'none');a.text=t;a.text.style={typeface:'Helvetica Neue',fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none'};return a;}
function line(pts,col=C.line,width=1.5,arrow=false){let xs=pts.map(v=>v[0]),ys=pts.map(v=>v[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(1,Math.max(...xs)-x),h=Math.max(1,Math.max(...ys)-y);s.shapes.add({geometry:'custom',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:col,width},customPaths:[{width:w,height:h,commands:pts.map((v,i)=>({[i?'lineTo':'moveTo']:{x:v[0]-x,y:v[1]-y}}))}]});if(arrow){let [ex,ey]=pts.at(-1);s.shapes.add({geometry:'custom',position:{left:ex-10,top:ey-6,width:10,height:12},fill:col,line:{fill:col,width:.1},customPaths:[{width:10,height:12,commands:[{moveTo:{x:0,y:0}},{lineTo:{x:10,y:6}},{lineTo:{x:0,y:12}},{close:{}}]}]});}}
async function icon(n,x,y,size,col=C.ink){let svg=(await fs.readFile(path.join(O,'assets',n+'.svg'),'utf8')).replaceAll('currentColor',col);s.images.add({blob:Buffer.from(svg),contentType:'image/svg+xml',alt:n+' icon, Lucide ISC license',fit:'contain',position:{left:x,top:y,width:size,height:size}});}
text('Company-focused news for four-hour prediction',48,30,1184,57,34,C.ink,true);
text('PROPOSED FULL-TEXT EXTENSION',50,99,900,27,15,C.muted,true);
// Three numbered headings establish one reading direction.
for(const [n,t,x] of [[1,'Read company passages',50],[2,'Group repeated content',460],[3,'Predict with context',895]]){text(String(n).padStart(2,'0'),x,155,46,38,23,C.teal,true);text(t,x+48,155,n===3?286:330,38,23,C.ink,true);}
// A: source documents, with content-level highlights.
text('Illustrative Apple reports',50,208,330,28,17,C.muted);
box(50,248,320,122,'#FFFFFF',C.line,6);await icon('file-text',64,262,26,C.muted);text('Report A',102,258,230,31,19,C.muted,true);
box(65,307,289,44,C.pale);text('Target price cut to $180',77,308,270,42,21,C.blue,true);
box(50,390,320,178,'#FFFFFF',C.line,6);await icon('file-text',64,404,26,C.muted);text('Report B',102,400,230,31,19,C.muted,true);
box(65,449,289,44,C.pale);text('Target price cut to $180',77,450,270,42,21,C.blue,true);
box(65,503,289,44,C.mint);text('Sales guidance lowered',77,504,270,42,21,C.teal,true);
text('Keep context, numbers and negation',50,584,330,28,17,C.muted);
// B: shared encoder and two content groups.
await icon('scan-text',463,218,36,C.ink);text('FinBERT',510,211,270,30,23,C.ink,true);text('Financial text → model features',510,242,300,29,17,C.muted);
box(460,303,338,96,C.pale,'none',6);await icon('layers',477,322,38,C.blue);text('Same target-price cut',530,313,255,33,22,C.blue,true);text('2 reports → 1 group summary',530,349,255,29,17,C.blue);
box(460,426,338,96,C.mint,'none',6);await icon('file-text',479,449,34,C.teal);text('Additional guidance',530,437,255,33,22,C.teal,true);text('Retained as a separate group',530,474,255,28,17,C.teal);
text('Average within each group,\nthen across groups',462,544,330,61,20,C.ink,false,'center');
// Two simple arrows, without crossed branches.
line([[389,415],[438,415]],C.muted,2,true);line([[818,415],[872,415]],C.muted,2,true);
// C: compact feature inputs and one trainable predictor.
await icon('layers',898,225,31,C.teal);text('Body summary',942,219,288,45,21,C.ink,true);
await icon('file-text',898,286,29,C.muted);text('Title summary',942,281,286,38,20);
await icon('layers',898,339,29,C.muted);text('Counts, sources, timing',942,334,286,38,20);
await icon('chart-no-axes-combined',898,392,31,C.muted);text('Historical prices',942,387,286,38,20);
line([[915,441],[915,461],[1204,461],[1204,441]],C.line,1.4);
box(930,485,265,57,C.ink,'none',6);text('Small trained classifier',938,493,249,40,21,'#FFFFFF',true,'center');
line([[1063,546],[1063,562]],C.muted,2);text('4-hour UP / DOWN',907,568,313,37,25,C.ink,true,'center');
line([[50,637],[1230,637]],C.line,1);
text('Repeated content is merged; additional details are preserved.',50,651,1180,39,23,C.ink,true);
s.speakerNotes.textFrame.setText('Proposed, not yet evaluated. Read company passages with context; group near-duplicates while retaining additional details. Frozen FinBERT encodes the text. Average within groups, then across groups. Combine with titles, reporting metadata and past prices in logistic regression. Missing body retains titles; no news uses price-only fallback. Example is illustrative. Icons: Lucide, ISC license, https://lucide.dev; github.com/lucide-icons/lucide.');
await (await PresentationFile.exportPptx(p)).save(path.join(W,'candidate.pptx'));
const {finalizePresentation}=await import(path.join(SKILL,'container_tools/artifact_tool_utils.mjs'));
console.log(await finalizePresentation({workspaceDir:ROOT,candidatePath:path.join(W,'candidate.pptx'),finalPath:path.join(O,'Slide6_FullText_Method.pptx'),pythonExecutable:'/Users/victor/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],fontPolicy:{basis:'design',families:['Helvetica Neue']},explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:path.join(W,'validation_final.json')}));
