import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';
const ROOT='/Users/victor/Documents/Codex/2026-09-14/wox';
const SKILL='/Users/victor/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11814/skills/presentations';
const OUT=path.join(ROOT,'outputs/teacher_meeting_20260921');
const BUILD=path.join(ROOT,'work/presentation_meeting/build/v10');
const SOURCE=path.join(OUT,'CSE573_News_Context_Faculty_Meeting_v9.pptx');
const FONT='Helvetica Neue', NAVY='#17324D',TEAL='#006E73',INK='#293744',MUTED='#536676',PALE='#E7F4F1',BLUE='#EDF3F9';
const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const originals=[...p.slides.items], slides=[], plans=[];
const repo='https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/';
const evidence=repo+'outputs/stock_paper_methods_4h/v1/REPORT.md';
const code=repo+'outputs/stock_paper_methods_4h/run.py';
const attribution=repo+'outputs/stock_combination_4h/v1/REPORT.md';
function text(s,t,x,y,w,h,size=26,color=INK,bold=false){let a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:FONT,fontSize:size,color,bold,autoFit:'none'};return a;}
function slide(title,notes){let s=originals[1].duplicate();for(const q of [...s.shapes.items])s.shapes.deleteById(q.id);let n=slides.length+1;text(s,title,42,33,1196,96,34,NAVY,true);text(s,String(n),1185,674,55,24,12,MUTED);s.speakerNotes.textFrame.setText(notes);slides.push(s);plans.push({title,notes});return s;}
function foot(s,t){text(s,t,42,600,1196,62,20,MUTED);}
function box(s,t,x,y,w,h,accent=false){let a=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:accent?PALE:BLUE,line:{fill:accent?'#88B7B1':'#B7C9DB',width:1}});a.text=t;a.text.style={typeface:FONT,fontSize:25,color:accent?TEAL:NAVY,bold:true,alignment:'center',verticalAlignment:'middle',autoFit:'none'};return a;}
function link(s,a,b,from='right',to='left'){s.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,line:{fill:'#627C8B',width:2},tail:{type:'arrow',width:'med',length:'med'}});}
function cols(title,left,right,notes){const s=slide(title,notes);for(const [i,item] of [left,right].entries()){const x=42+i*615;text(s,item[0],x,175,555,66,29,NAVY,true);text(s,item[1],x,255,555,310,27);}return s;}
function table(s,values,{top=190,widths=null,rowHeight=60,font=23,highlight=-1}={}){const t=s.tables.add({rows:values.length,columns:values[0].length,left:42,top,width:1196,height:values.length*rowHeight,...(widths?{columnWidths:widths}:{}),values});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){let z=t.getCell(r,c);z.fill=r===0?NAVY:r===highlight?PALE:r%2? '#FFFFFF':BLUE;z.text.style={typeface:FONT,fontSize:font,color:r===0?'#FFFFFF':INK,bold:r===0||r===highlight};}t.borders.assign({fill:'#D4E0E7',width:.6,style:'solid'});return t;}
let s=slide('','Opening: Our implemented method represents financial news with frozen FinBERT, groups near-duplicate titles, retains coverage/arrival metadata, and trains a small price-plus-news classifier. This deck replaces the former relative-fact-change proposal as the main presentation. The two methods are distinct. Sources: '+code+' ; '+evidence);
text(s,'CSE 573  /  Faculty discussion  /  21 September 2026',42,48,1196,44,24,TEAL);
text(s,'Grouping financial news\nfor stock prediction',42,220,1160,190,69,NAVY);
text(s,'FinBERT, reporting patterns and historical prices\nApple and Amazon · Four-hour direction',42,466,1150,98,28);

s=slide('The prediction task','Canonical task: 1,607 total windows including warmup; 765 training forward predictions, 252 development, 357 later. Cutoff is five minutes before the four-hour target. Only complete available price history and admitted news are inputs. All evaluation periods are exposed exploratory historical backtests. Source: '+attribution);
text(s,'Given news and prices already available, predict whether the next four-hour window rises or falls.',42,155,1190,86,31,NAVY,true);
let a=box(s,'Available news\nand completed prices',42,310,340,120),b=box(s,'Prediction cutoff\n5 min before start',447,310,310,120),c=box(s,'Target window\nNext 4 hours',822,310,416,120);link(s,a,b);link(s,b,c);
foot(s,'1,607 AAPL / AMZN windows. Existing periods are exploratory historical backtests.\nApple and Amazon are the companies; AAPL and AMZN are their stock tickers.');

s=slide('Baseline: average article meaning, then predict','The semantic reference uses frozen ProsusAI/FinBERT title embeddings, train-article PCA16, 16 old price features and basic news metadata in a regularized logistic classifier. No admitted news falls back to saved R1. The matched grouping control later has the same six metadata fields. Source: '+code+' ; '+evidence+' . FinBERT background: https://arxiv.org/abs/1908.10063');
text(s,'FinBERT: a language model trained on financial text that converts a title into numerical features.',42,155,1190,76,28);
a=box(s,'Each available\nnews title',42,290,235,108);b=box(s,'Frozen\nFinBERT',322,290,235,108);c=box(s,'Average all\narticle vectors',602,290,280,108);let d=box(s,'Classifier\n+ price history',927,290,311,108);link(s,a,b);link(s,b,c);link(s,c,d);
foot(s,'The encoder stays fixed. The classifier learns from labeled historical windows.\nThe same story can influence the average several times when several outlets report it.');

s=slide('Why repeated reports can dominate the input','Illustrative teaching example, not six observed source articles. Five compatible near-duplicate Apple target-price-cut reports and one distinct product report. The example is designed to explain article weighting, not predict a market return. Equal group weighting is a modeling hypothesis, not verified event importance.');
text(s,'One window contains six titles',42,154,1196,54,30,NAVY,true);
table(s,[['News content','Reports','Share of article average'],['Apple target price cut','5 near-duplicates','5/6 = 83.3%'],['Apple introduces a product','1 distinct report','1/6 = 16.7%']],{top:246,widths:[590,245,361],rowHeight:78,font:27});
foot(s,'Illustration only. Counting every article gives the repeated story most of the representation.\nWe test whether grouping improves this representation while retaining reporting volume separately.');

s=slide('Our method: group content and retain reporting patterns','This is the implemented N1M structure, described without internal IDs. Accepted input is titles, not whole-body event extraction. FinBERT title vectors are averaged within complete-link compatible near-title groups, then across groups; PCA16 fits training articles only. Six metadata columns and 16 OLD price columns enter regularized LR. No news returns R1. Source: '+code);
a=box(s,'Available titles\nfor the stock',42,220,235,108);b=box(s,'FinBERT +\nreport grouping',322,220,250,108,true);c=box(s,'Group summary\n+ metadata',617,220,275,108,true);d=box(s,'Small classifier\nUp probability',937,220,301,108);link(s,a,b);link(s,b,c);link(s,c,d);
let e=box(s,'Report counts, sources\nand arrival ages',617,430,275,105);link(s,e,c,'top','bottom');let f=box(s,'Completed\nprice history',937,430,301,105);link(s,f,d,'top','bottom');
foot(s,'Two-step averaging reduces duplicate weight. Metadata preserves how widely and recently the story appeared.');

s=slide('How we form a group','Exact implementation: title tokens are lowercase alphanumeric sets; Jaccard >= 0.8; guards require equal raise/lower/maintain/initiate/deny/accuse patterns, numerical tokens and quarter expressions. Each new title must be compatible with every group member. Process admitted articles available by the cutoff. This is near-report grouping, not validated event coreference. Source: '+code);
text(s,'Compare titles already available by the prediction cutoff',42,152,1190,55,29,NAVY,true);
text(s,'1',42,260,50,60,38,TEAL,true);text(s,'Mostly the same words',110,261,465,60,29,NAVY,true);text(s,'At least 80% overlap in their word sets.',110,323,465,90,26);
text(s,'2',659,260,50,60,38,TEAL,true);text(s,'Compatible factual cues',727,261,495,60,29,NAVY,true);text(s,'Keep different numbers, action cues\nor reporting quarters apart.',727,323,495,105,26);
text(s,'Every title must match all other titles in its group.',110,483,1080,52,29,TEAL,true);
foot(s,'“Target cut to 180” and “target cut to 160” stay separate.\nSimilar titles suggest near-duplicate reporting; they do not establish that two articles describe the same event.');

s=slide('Two-step averaging changes the balance','Illustration follows the six-report example. Every title first has a vector. Average the five cut-report vectors into c1 and the single product vector into c2, then compute (c1+c2)/2. Before PCA this gives each group half the aggregate weight and each of the five reports 1/10. This does not mean the classifier assigns equal causal importance to the two stories.');
a=box(s,'5 similar cut reports',42,205,330,94);b=box(s,'Mean of group A',440,205,340,94,true);link(s,a,b);c=box(s,'1 product report',42,370,330,94);d=box(s,'Mean of group B',440,370,340,94,true);link(s,c,d);e=box(s,'Average the\n2 group vectors',864,282,374,110,true);link(s,b,e);link(s,d,e);
text(s,'Article average: 83.3% / 16.7%       Group average: 50% / 50%',42,517,1196,56,29,NAVY,true);
foot(s,'Illustration only. Equal group weights are a simple testable choice, not a claim of equal economic importance.');

s=slide('Metadata preserves the reporting pattern','N0M and N1M use exactly these six fields. Actual inputs apply log1p to the five count/age values; only_duplicates is binary and means every group contains more than one article. Illustrative six-article arrival ages 10,20,30,50,60,70 minutes have newest age10 and median40; five duplicates plus one singleton gives false. Availability is the source-defined arrival time, not certified first market disclosure. Sources count recognizable sites, not independent evidence. Source: '+code);
table(s,[['Field','Six-report example','What the learner receives'],['Articles / groups / sources','6 / 2 / 3','Volume and diversity of reporting'],['Newest / median arrival age','10 min / 40 min','How recently reports arrived'],['All groups contain repeats?','No','Whether every group has duplicates']],{top:214,widths:[435,278,483],rowHeight:77,font:25});
foot(s,'Metadata means descriptive information about the reports. Counts stay available after content grouping.\nArrival age measures when we received a report, not when the market first learned the fact.');

s=slide('How news and prices reach the forecast','N1M uses 16 OLD price variables (six return/range pairs, history age, mean return, return std, NY hour), PCA16 of group means, six metadata columns. Price normalization and semantic/metadata scaling fit training only. LogisticRegression liblinear C grid .01,.1,1, max_iter3000 tol1e-7 seed573; no news replaces the joint probability with R1. Do not conflate the predictor with generative LLM or state that recent extra R1 columns are in the N1M joint training. Source: '+code);
a=box(s,'16 news features\nCompressed group meaning',42,175,475,100,true);b=box(s,'6 reporting features\nCounts and arrival ages',42,325,475,100,true);c=box(s,'16 price features\nReturns, ranges and timing',42,475,475,100);d=box(s,'Regularized logistic regression\nLearns a weight for each feature',666,260,572,122);link(s,a,d);link(s,b,d);link(s,c,d);e=box(s,'Four-hour upward probability',756,467,482,100);link(s,d,e,'bottom','top');
foot(s,'Regularization limits large fitted weights. With no admitted news, the system returns its price-only forecast.');

s=cols('Training uses earlier months to predict later months',['What learns from data','FinBERT stays frozen.\n\nFit compression and scaling using training data. Learn the small classifier from historical up/down labels.'],['How settings are chosen','Choose regularization using earlier monthly validation results.\n\nFreeze the final model at August 2018. Report development and later periods separately.'],'March–August monthly issued C choices use previous CV records; March uses default .1. Final training before September chooses C from preceding monthly CV; final evaluation September onward. Code asserts training label end before evaluation cutoff. Inner raw classifier metrics select C and R1 fallback applies to issued no-news rows. This distinction is recorded rather than implying a joint fallback selection objective. Sources: '+code+' ; '+evidence);
foot(s,'Existing development and later periods have already informed project exploration.\nThese chronological historical results are not a new untouched test.');

s=slide('Matched comparisons isolate the mechanism','Reference F2, N0M, N1, N1M all use the frozen title representation and OLD price features, canonical windows, training selection framework and R1 no-news fallback. Basic news metadata differs from six-field expanded metadata. Main grouping attribution comparison is N1M vs N0M; different selected C values remain a source of combined representation/regularization effects. Source: '+code+' ; '+evidence);
table(s,[['Method','Content weighting','Reporting metadata'],['FinBERT reference','Each article equally','Basic count / availability'],['Matched control','Each article equally','Same six fields'],['Grouping only','Each group equally','Basic count / availability'],['Our complete method','Each group equally','Same six fields']],{top:188,widths:[414,386,396],rowHeight:70,font:25,highlight:4});
foot(s,'The key comparison is complete method vs matched control. It changes aggregation while retaining metadata.');

s=slide('Historical results: gains differ by stock and period','All cells read from stock_paper_methods_4h/v1/metrics.csv. F2, N0M, N1, N1M. BA percentages rounded to two decimals. Development September–October2018, later November2018–February2019. All are exposed exploratory historical results. N1M vs N0M later AAPL -0.58pp and AMZN +2.58pp. Source: '+evidence);
text(s,'Balanced accuracy (%)   Development: Sep–Oct 2018   Later: Nov 2018–Feb 2019',42,155,1196,60,23,MUTED);
table(s,[['Method','AAPL dev','AAPL later','AMZN dev','AMZN later'],['FinBERT reference','54.13','56.71','46.29','54.27'],['Matched control','52.54','56.74','47.59','54.74'],['Grouping only','52.79','54.59','48.14','56.79'],['Our complete method','51.80','56.16','50.46','57.32']],{top:235,widths:[380,204,204,204,204],rowHeight:60,font:25,highlight:4});
foot(s,'Later change vs matched control: AAPL −0.58 pp, AMZN +2.58 pp.\nBA balances correct up/down recognition. All cells are exposed exploratory historical backtests.');

s=cols('What the result does and does not explain',['Observed','AMZN later: 8 errors repaired,\n4 new errors introduced.\n\nMean training-month BA change:\nAAPL −0.64 pp, AMZN +0.12 pp.'],['Interpretation','The complete method can change predictions, but the gain is not consistent across periods and stocks.\n\nStronger regularization may explain part of the AMZN gain.'],'N1M vs N0M mean March–August monthly BA AAPL -.64pp AMZN+.12pp macro-.26pp, failed original promotion. Later AMZN repaired8 introduced4. Article training C1 vs group training C.01. Fixed-model replacement of article/group current vectors changed no directions; therefore no proof duplicate removal caused the gain. BA paired one-day interval [-.08,5.48]pp and five-day[-.24,6.14]pp cross0. Source: '+evidence+' ; '+attribution);
foot(s,'The training-period improvement rule was not met.\nAn AMZN later-period gain supports further diagnosis, not a stable two-stock improvement claim.');

s=cols('The course contribution is how we organize information',['Web mining','Find near-duplicate financial reports using title similarity and factual cues.\n\nKeep content redundancy separate from reporting volume and arrival time.'],['Machine learning','Use financial-language features alongside price history.\n\nTest the grouping mechanism against a control with the same reporting metadata.'],'Course alignment: semantic representation, similarity-based document grouping, source/time metadata mining and classical supervised prediction. This method does not establish an entity-event knowledge graph and does not require LLM generation or extraction labels. No claim that the instructor has endorsed this particular mechanism. Source: '+code);
foot(s,'The contribution under test is the news representation and aggregation rule.\nFinBERT supplies the text features; it is not itself our new contribution.');

s=cols('Current method and next decision',['Implemented system','Frozen FinBERT + near-report groups\n+ six metadata fields + price features\n+ a small trained classifier.\n\nSame architecture for both stocks.'],['Evidence and next check','AMZN improves in one later period.\nAAPL and earlier months are mixed.\n\nA fixed-regularization comparison would separate grouping from model shrinkage.'],'This presentation is a mechanism explanation of the existing implemented N1M system. It does not run new training or authorize restart of paused experiments. The full current-vs-prior fact-change reader is optional future work rather than the claimed implemented mechanism. Group membership is heuristic, and entity review is not claimed. Source: '+attribution);
foot(s,'Discussion: does reducing repeated content while preserving reporting patterns make a useful course contribution?');

s=slide('Appendix: exact representation and model','Technical detail from run.py. Title Jaccard .8, complete-link compatibility, group and article means. PCA learned on distinct training articles, no groups receive a learned attention weight. log1p metadata features. 16 OLD price features. Fixed regularization candidate grid .01,.1,1. No-news fallback is saved R1 rather than zeroed joint output. '+code);
text(s,'Group summary = mean of the title vectors within that group',42,165,1196,58,29,NAVY,true);
text(s,'Window summary = mean of the group summaries',42,237,1196,58,29,TEAL,true);
table(s,[['Component','Implementation'],['Semantic compression','PCA to 16 features, fitted on training articles'],['Classifier input','16 news + 6 metadata + 16 price features'],['Classifier / regularization','Logistic regression / C = 0.01, 0.1 or 1'],['No admitted news','Return the saved price-only probability']],{top:333,widths:[395,801],rowHeight:48,font:23});

s=slide('Appendix: a repaired error and an introduced error','Saved case descriptions are from combination report and do not disclose raw full article text. AMZN Jan8 before .606 after .499 actual down; Jan31 before .621 after .493 actual up. Both have distinct reports; Jan8 has2articles2groups and Jan31 two distinct articles. These are rounded saved probabilities, not new inference. They illustrate retraining changes, not causal news impact. '+attribution);
table(s,[['AMZN example','Article model','Group model','Actual result'],['8 Jan 2019: valuation / competition','60.6% up','49.9% up','Down: repaired'],['31 Jan 2019: earnings / partnership','62.1% up','49.3% up','Up: introduced error']],{top:200,widths:[500,215,215,266],rowHeight:86,font:24});
text(s,'Both examples contain distinct reports.',42,495,1196,50,30,NAVY,true);
foot(s,'Training and selected regularization changed between models.\nThese cases cannot be explained as simply removing a duplicate from the current window.');

s=cols('Appendix: sources and scope',['Method and code','Araci (2019): FinBERT.\n\nProject implementation:\nstock_paper_methods_4h/run.py\n\nSaved results and matched controls:\nstock_paper_methods_4h/v1/REPORT.md'],['Interpretation','AMZN attribution study:\nstock_combination_4h/v1/REPORT.md\n\nTitle grouping is heuristic.\nA modern encoder on 2018 news is a retrospective study.'],'FinBERT: https://arxiv.org/abs/1908.10063 . Implementation: '+code+' . Results: '+evidence+' . Attribution: '+attribution+' . Internal IDs: reference F2, matched N0M, grouping-only N1, complete N1M. No new model fit or verification resumes during this deck edit. All historical slide versions preserved.');
foot(s,'Exact source links and explanations appear in the speaker notes. No relative-fact reader or graph model is claimed here.');

for(const o of originals)o.delete();
for(let i=0;i<slides.length;i++)slides[i].moveTo(i);
await fs.mkdir(BUILD,{recursive:true});
await fs.writeFile(path.join(BUILD,'storyboard.json'),JSON.stringify(plans,null,2));
await (await PresentationFile.exportPptx(p)).save(path.join(BUILD,'candidate.pptx'));
await fs.mkdir(path.join(BUILD,'renders'),{recursive:true});
for(let i=0;i<slides.length;i++){const b=await p.export({slide:slides[i],format:'png',scale:1});await fs.writeFile(path.join(BUILD,'renders',`slide-${i+1}.png`),new Uint8Array(await b.arrayBuffer()));}
await fs.writeFile(path.join(OUT,'PRESENTER_NOTES_v10.md'),'# Grouping financial news: presenter notes\n\n15 main slides and 3 appendix slides. Suggested speaking time: 10–12 minutes.\n\n'+plans.map((x,i)=>`## Slide ${i+1}: ${x.title||'Grouping financial news for stock prediction'}\n\n${x.notes}\n`).join('\n'));
const {finalizePresentation}=await import(path.join(SKILL,'container_tools/artifact_tool_utils.mjs'));
const finalPath=path.join(OUT,'CSE573_Grouped_News_Faculty_Meeting_v10.pptx');
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:path.join(BUILD,'candidate.pptx'),finalPath,pythonExecutable:'/Users/victor/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[4,8,11,12,16,17].flatMap(n=>['--require-native-table-slide',String(n)])],fontPolicy:{basis:'reference',families:[FONT],referencePath:SOURCE,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(SOURCE)).digest('hex')},explicitTotalSlideCount:18,requiredNativeTableOwnerSlides:[4,8,11,12,16,17],verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')});
console.log(JSON.stringify(result,null,2));
