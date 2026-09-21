import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const ROOT='/Users/victor/Documents/Codex/2026-09-14/wox';
const BUILD=path.join(ROOT,'work/presentation_meeting/fulltext_v1');
const FIG=path.join(ROOT,'outputs/teacher_meeting_20260921/fulltext_figure_v1');
const SOURCE=path.join(BUILD,'before_edit.pptx');
const FONT='Helvetica Neue',MUTED='#536676';
const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const originals=[...p.slides.items];
const s=originals[5].duplicate();
for(const q of [...s.shapes.items])s.shapes.deleteById(q.id);
for(const q of [...s.images.items])s.images.deleteById(q.id);
function text(t,x,y,w,h,size=24,color='#17324D',bold=false){const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=t;q.text.style={typeface:FONT,fontSize:size,color,bold,autoFit:'none'};}
async function paperFigure(s,name){
 const data=JSON.parse(await fs.readFile(path.join(FIG,name+'.json'),'utf8'));const k=1196/data.w,ox=42,oy=105;
 function pathShape(pts,fill,line,closed=false){const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);const x=Math.min(...xs),y=Math.min(...ys),w=Math.max(1,Math.max(...xs)-x),h=Math.max(1,Math.max(...ys)-y);const commands=pts.map((p,i)=>({[i?'lineTo':'moveTo']:{x:(p[0]-x)*k,y:(p[1]-y)*k}}));if(closed)commands.push({close:{}});return s.shapes.add({geometry:'custom',position:{left:ox+x*k,top:oy+y*k,width:w*k,height:h*k},fill,line,customPaths:[{width:w*k,height:h*k,commands}]});}
 for(const e of data.edges){pathShape(e.pts,'none',{fill:MUTED,width:1.7,style:e.dash?'dashed':'solid'});const [x,y]=e.pts.at(-1),[px,py]=e.pts.at(-2),a=Math.atan2(y-py,x-px);pathShape([[x,y],[x-13*Math.cos(a)+6*Math.sin(a),y-13*Math.sin(a)-6*Math.cos(a)],[x-13*Math.cos(a)-6*Math.sin(a),y-13*Math.sin(a)+6*Math.cos(a)]],MUTED,{fill:MUTED,width:.1},true);}
 for(const n of data.nodes){if(n.id==='panel')continue;const q=s.shapes.add({geometry:n.textOnly?'textbox':'rect',position:{left:ox+n.x*k,top:oy+n.y*k,width:n.w*k,height:n.h*k},fill:n.textOnly?'none':n.fill,line:{fill:n.textOnly?'none':n.stroke,width:n.textOnly?0:1.4},...(n.textOnly?{}:{borderRadius:5})});q.text=n.text;q.text.style={typeface:FONT,fontSize:n.font*k,color:n.color,bold:n.bold,alignment:n.align==='left'?'left':'center',verticalAlignment:'middle',autoFit:'none'};}
}
text('Proposed method: company-focused full text',42,30,1196,54,34,'#17324D',true);
await paperFigure(s,'fulltext_method');
text('Design proposal; not evaluated here. LR = logistic regression. Fit preprocessing and classifier on past data only.',42,669,1160,28,17,'#536676');
text('6',1205,675,30,22,12,'#536676');
s.speakerNotes.textFrame.setText('Proposed full-text extension, not implemented or evaluated in this slide task. Existing reported scores apply to the earlier headline pipeline, not this design. All news is available by the prediction cutoff. Select target-company passages using explicit aliases and surrounding context; preserve negation, numbers, sentence IDs, and unknown flags. Build sentence-bounded chunks within token budget. Group near-duplicate passage content using conservative similarity and numeric/action/period guards; retain unmatched details without claiming market-first novelty. FinBERT stays frozen. Encode titles separately from company passages; within each passage group average member embeddings, then average group summaries. Keep title and body summaries separate for training-only compression/scaling. Reporting features use raw metadata and selection/group outcomes. Historical completed prices join the classifier directly. Missing reliable body retains title; no admitted news uses price-only fallback. Numerical feature dimensions, final chunk budget, similarity thresholds, classifier settings and quality thresholds require preregistration; this figure does not invent them. Illustration: one report covers a price target cut to180; another repeats it and adds guidance; repeated passages form one group while guidance remains separate. No new predictive experiment was run.');
for(const old of originals)old.delete();
await (await PresentationFile.exportPptx(p)).save(path.join(BUILD,'candidate.pptx'));
const SKILL='/Users/victor/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11814/skills/presentations';
const {finalizePresentation}=await import(path.join(SKILL,'container_tools/artifact_tool_utils.mjs'));
const res=await finalizePresentation({workspaceDir:ROOT,candidatePath:path.join(BUILD,'candidate.pptx'),finalPath:path.join(FIG,'Slide6_FullText_Proposal_v2.pptx'),pythonExecutable:'/Users/victor/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],fontPolicy:{basis:'reference',families:[FONT],referencePath:SOURCE,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(SOURCE)).digest('hex')},explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation_v2.json')});
console.log(JSON.stringify(res));
