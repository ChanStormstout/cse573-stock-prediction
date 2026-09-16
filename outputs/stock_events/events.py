"""Frozen headline-only primary event classifier; no outcome labels or external facts."""
import re,html
EVENTS=['earnings','guidance','analyst_rating','product_business','legal_regulatory','holdings','market_recap','other','unknown']
ALIAS={'AAPL':r'\b(?:Apple|AAPL)\b','AMZN':r'\b(?:Amazon(?:\.com)?|AMZN)\b'}
DEFINITIONS={
'earnings':'A company quarterly/annual financial results announcement or preview, not P/E valuation background.',
'guidance':'Management issued or revised a future financial forecast. Analyst estimates and investor shareholding changes are not management guidance.',
'analyst_rating':'An analyst rating or price-target action; ordinary product commentary without a rating action is not a rating.',
'product_business':'Products, services, supply/production, commercial partnerships, sales events or operating developments.',
'legal_regulatory':'Lawsuits, courts, crimes or regulatory actions.',
'holdings':'An investor bought, sold, increased or reduced ownership of TARGET shares, or expressed an ownership preference.',
'market_recap':'Primary news is a stock-price move, valuation milestone or market/stock investment commentary. Mentioned catalysts can be secondary.',
'other':'A clear event outside these categories, e.g. personnel or labor conditions.',
'unknown':'Not enough relevant information about TARGET to identify an event.'}
def target_clause(title,symbol):
 parts=re.split(r';|\s[|]\s',html.unescape(title))
 picked=[p.strip() for p in parts if re.search(ALIAS[symbol],p,re.I)]
 return ' ; '.join(picked) if picked else html.unescape(title)
def rule_event(title,symbol):
 s=target_clause(title,symbol)
 if not re.search(ALIAS[symbol],s,re.I):return 'unknown',s
 tests=[
 ('holdings',r'\b(?:stake|shareholder|holder|holding|holdings|position)\b|buffett.*(?:own|love)|(?:buy|sell|buying|selling)\s+(?:more\s+)?(?:apple|amazon|aapl|amzn)\s+(?:shares|stock)'),
 ('analyst_rating',r'price target|target price|upgrad|downgrad|rating.{0,30}(?:reaffirm|reiterat|maintain)|(?:reaffirm|reiterat|maintain).{0,30}rating|initiated with'),
 ('guidance',r'\bguidance\b|(?:raises|lowers|cuts|boosts|revises).{0,20}(?:revenue|profit|earnings)\s+(?:forecast|outlook)'),
 ('market_recap',r'\b(?:stock|stocks|shares)\b.{0,35}\b(?:hits?|slides?|slips?|jumps?|falls?|rises?|surges?|dips?|rallies|tops?|plunges?)\b|\b(?:slide|market cap|market capitalization|bullish|bearish|market valuation)\b'),
 ('earnings',r'\bearnings\b|quarterly results|reports?\s+(?:record\s+)?(?:profit|loss)'),
 ('legal_regulatory',r'lawsuit|\bsued\b|\bcourt\b|antitrust|\bprison\b|\bfraud\b|patent.{0,30}(?:battle|dispute|infring|suit)'),
 ('product_business',r'launch|unveil|prime day|iphone|homepod|alexa|\bchip\b|streaming|service|product|partnership|checking account|cobalt|smart speaker|music|sales|production|delivery')]
 for event,pat in tests:
  if re.search(pat,s,re.I):return event,s
 return 'other',s
