import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const ROOT='/Users/victor/Documents/Codex/2026-09-14/wox';
const BUILD=path.join(ROOT,'work/presentation_meeting/reference_fulltext_v4');
const FIG=path.join(ROOT,'outputs/teacher_meeting_20260921/reference_fulltext_v4');
const SOURCE=path.join(BUILD,'before_edit.pptx');
const FONT='Helvetica Neue',MUTED='#607181';
const p=Presentation.create({slideSize:{width:1280,height:720}});const s=p.slides.add();s.background.fill='#FFFFFF';
function text(t,x,y,w,h,size=24,color='#17324D',bold=false){const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=t;q.text.style={typeface:FONT,fontSize:size,color,bold,autoFit:'none'};}
async function paperFigure(s,name){
 const data=JSON.parse(await fs.readFile(path.join(FIG,name+'.json'),'utf8'));const k=1196/data.w,ox=42,oy=145;
 function pathShape(pts,fill,line,closed=false){const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);const x=Math.min(...xs),y=Math.min(...ys),w=Math.max(1,Math.max(...xs)-x),h=Math.max(1,Math.max(...ys)-y);const commands=pts.map((p,i)=>({[i?'lineTo':'moveTo']:{x:(p[0]-x)*k,y:(p[1]-y)*k}}));if(closed)commands.push({close:{}});return s.shapes.add({geometry:'custom',position:{left:ox+x*k,top:oy+y*k,width:w*k,height:h*k},fill,line,customPaths:[{width:w*k,height:h*k,commands}]});}
 for(const e of data.edges){pathShape(e.pts,'none',{fill:MUTED,width:1.7,style:e.dash?'dashed':'solid'});const [x,y]=e.pts.at(-1),[px,py]=e.pts.at(-2),a=Math.atan2(y-py,x-px);pathShape([[x,y],[x-13*Math.cos(a)+6*Math.sin(a),y-13*Math.sin(a)-6*Math.cos(a)],[x-13*Math.cos(a)-6*Math.sin(a),y-13*Math.sin(a)+6*Math.cos(a)]],MUTED,{fill:MUTED,width:.1},true);}
 for(const n of data.nodes){if(n.id==='panel')continue;const q=s.shapes.add({geometry:n.textOnly?'textbox':'rect',position:{left:ox+n.x*k,top:oy+n.y*k,width:n.w*k,height:n.h*k},fill:n.textOnly?'none':n.fill,line:{fill:n.textOnly?'none':n.stroke,width:n.textOnly?0:1.4},...(n.textOnly?{}:{borderRadius:5})});q.text=n.text;q.text.style={typeface:FONT,fontSize:n.font*k,color:n.color,bold:n.bold,alignment:n.align==='left'?'left':'center',verticalAlignment:'middle',autoFit:'none'};}
}
text('Our proposed method: full text + reporting patterns',42,30,1196,64,34,'#1E314B',true);
await paperFigure(s,'method');
s.speakerNotes.textFrame.setText('Proposed full-text extension, not a new experimental result. Prepared records retain original titles, target-company body passages and report provenance. The same frozen FinBERT encodes titles and passages separately. Conservative passage groups preserve unmatched details and separate conflicting numeric/action cues. Group IDs govern body averaging only: within groups, then across groups. Titles are averaged separately. Join text summaries, reporting/coverage features and completed historical-price features in trained logistic regression. Missing body retains titles; no news uses price-only fallback. Dimensions and thresholds require preregistration.');
await (await PresentationFile.exportPptx(p)).save(path.join(BUILD,'candidate.pptx'));
const SKILL='/Users/victor/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11814/skills/presentations';
const {finalizePresentation}=await import(path.join(SKILL,'container_tools/artifact_tool_utils.mjs'));
const res=await finalizePresentation({workspaceDir:ROOT,candidatePath:path.join(BUILD,'candidate.pptx'),finalPath:path.join(FIG,'Slide6_FullText_Reference_Final.pptx'),pythonExecutable:'/Users/victor/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],fontPolicy:{basis:'design',families:[FONT]},explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation_final.json')});
console.log(JSON.stringify(res));
