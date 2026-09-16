# 40 个事件抽取检查案例

2026-09-15。前 20 个是此前模型已处理的诊断样本；后 20 个为新增检查样本。全部来自训练期质量开发集，未使用股票结果制定事件标签。标签为助手在本轮模型输出前对标题与所提供摘录的判断，未经组员独立复核；不是全文金标准。允许的多标签表示分类边界确有歧义，并非看到模型结果后放宽。

## 00 · AMZN · diagnostic_seen

**标题：** Foxconn Issues Investigation of Labor Conditions at China Factory Used for Amazon

- 记录：`2018_06_d157b48c57be246ec7dd80e7af4388a2.zip::news_0003138.json`
- 助手预先允许的事件：`other`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Foxconn Issues Investigation of Labor Conditions at China Factory Used for Amazon
- 诊断说明：Internal labor investigation; no government enforcement in supplied text.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Foxconn Issues Investigation of Labor Conditions at China Factory Used for Amazon
> 11 Jun, 2018 Tweet Amazon.com, Inc. (NASDAQ: AMZN) plant in Hengyang, China that makes Echo Dot smart speakers and Kindle e-readers, is under investigation by worlds largest Taiwan-based contract electronics manufacturer Foxconn that is formally known as Hon Hai Precision Industry Co Ltd (TPE: 2354), said on Sunday, due to reports of harsh, illegal, and inhumane conditions.
> “They were underpaid,” said China Labor Watch Program Officer Elaine Lu, “Thats illegal.” Amazon claimed they audited the factory in March and found overtime and use of dispatch workers were “issues of concern.” “We immediately requested a corrective action plan from Foxconn,” Amazon said in a statement.

## 01 · AAPL · diagnostic_seen

**标题：** Millennial investors are gobbling up shares of Apple ahead of earnings

- 记录：`2018_05_d157b48c57be246ec7dd80e7af4388a2.zip::news_0000382.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`earnings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S1`：in General Apple is scheduled to report second-quarter earnings after the closing bell Tuesday.Ahead of the report, millennial investors were buying shares 52% more than they were selling, according to Robinhood data.
- 诊断说明：Investor purchases are main event; earnings are context.

**完整的本次目标摘录（不代表完整新闻正文）：**

> in General Apple is scheduled to report second-quarter earnings after the closing bell Tuesday.Ahead of the report, millennial investors were buying shares 52% more than they were selling, according to Robinhood data.
> Follow - Apple - Stock - Price - Real-time Follow Apple's stock price in real-time here.Millennial investors seem to be expecting an post-earnings pop for Apple.
> Data - App - Robinhood - Users - Brokerages Data from stock-trading app Robinhood, whose users tend to skew much younger than traditional brokerages, show investors are snapping up shares of the tech giant 52% more than they are selling ahead of its earnings report after the closing bell Tuesday."AAPL significantly plunged since chip provider TSM announced disappointing earnings, blaming weak demand for an expensive smartphone," Sahill Poddar, the app's data scientist, told Business Insider in an email.

## 02 · AAPL · diagnostic_seen

**标题：** Apple (AAPL) is Thrivent Financial For Lutherans’ 6th Largest Position

- 记录：`2018_06_d157b48c57be246ec7dd80e7af4388a2.zip::news_0003353.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Apple (AAPL) is Thrivent Financial For Lutherans’ 6th Largest Position
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Thrivent Financial For Lutherans trimmed its stake in Apple (NASDAQ:AAPL) by 1.5% in the 4th quarter, according to its most recent filing with the Securities and Exchange Commission (SEC).
> Apple (NASDAQ:AAPL) last issued its quarterly earnings data on Tuesday, May 1st.
> Apple declared that its Board of Directors has approved a share buyback plan on Tuesday, May 1st that allows the company to repurchase $100.00 billion in shares.

## 03 · AMZN · diagnostic_seen

**标题：** Amazon scammers headed to federal prison after $1.2 million fraud

- 记录：`2018_06_d157b48c57be246ec7dd80e7af4388a2.zip::news_0002353.json`
- 助手预先允许的事件：`legal_regulatory`
- 简单规则：`legal_regulatory`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Amazon scammers headed to federal prison after $1.2 million fraud
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Unlike some shoppers who say they were cut off from their Amazon.com Inc. accounts after a few returns, an Indiana husband-and-wife team was able to defraud the e-commerce giant for years before getting nabbed by authorities.
> Between 2014 and 2017, the group repeatedly took advantage of Amazon’s AMZN, -0.13% customer service policy , claiming that items were damaged when they received them, and then requested free replacements.
> Amazon shares are up 67% for the past year while the S&P 500 index SPX, +0.18% is up nearly 14% for the period.

## 04 · AAPL · diagnostic_seen

**标题：** Apple (AAPL) Holder National Mutual Insurance Federation Of Agricultural Cooperatives Has Trimmed Position by $484,300 as Stock Price Rose; Forward Management Holds Stake in Blackstone Group LP (BX)

- 记录：`2018_05_d157b48c57be246ec7dd80e7af4388a2.zip::news_0008694.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `none`：无
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple (AAPL) Holder National Mutual Insurance Federation Of Agricultural Cooperatives Has Trimmed Position by $484,300 as Stock Price Rose; Forward Management Holds Stake in Blackstone Group LP (BX)
> Apple Inc. (NASDAQ:AAPL) has risen 22.56% since May 30, 2017 and is uptrending.
> AAPL’s profit will be $10.76B for 21.45 P/E if the $2.19 EPS becomes a reality.

## 05 · AMZN · diagnostic_seen

**标题：** Amazon's market cap on track to pass Microsoft for first time - MarketWatch

- 记录：`2018_02_d157b48c57be246ec7dd80e7af4388a2.zip::news_0001377.json`
- 助手预先允许的事件：`market_recap`
- 简单规则：`market_recap`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Amazon's market cap on track to pass Microsoft for first time - MarketWatch
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Amazon.com Inc.'s stock AMZN, +0.85% surged 0.8% in midday trade Wednesday, to lift its market value above Microsoft Corp. MSFT, -0.08% for the first time, according to data provided by WSJ Market Data Group.
> With a market capitalization of $704.38 billion, Amazon is now the third move valuable U.S. company, just above 4th-place Microsoft at $701.91 billion.
> Amazon is still far from 1st-place Apple Inc. AAPL, -0.93% which is valued at $819.71 billion.

## 06 · AAPL · diagnostic_seen

**标题：** Apple (AAPL) is said to develop displays to replace Samsung Electronics (005930 KS) screens – RedlionTrader

- 记录：`2018_03_d157b48c57be246ec7dd80e7af4388a2.zip::news_0004519.json`
- 助手预先允许的事件：`product_business`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `none`：无
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple (AAPL) is said to develop displays to replace Samsung Electronics (005930 KS) screens Posted By: RanSquawk March 18, 2018 

## 07 · AAPL · diagnostic_seen

**标题：** As Apple Computer (AAPL) Stock Price Declined, Rock Point Advisors Lowered Its Stake; Abbvie (ABBV) Market Value Declined While National Pension Service Upped Holding

- 记录：`2018_05_d157b48c57be246ec7dd80e7af4388a2.zip::news_0001417.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：As Apple Computer (AAPL) Stock Price Declined, Rock Point Advisors Lowered Its Stake; Abbvie (ABBV) Market Value Declined While National Pension Service Upped Holding
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Rock Point Advisors Llc decreased its stake in Apple Computer (AAPL) by 5.56% based on its latest 2017Q4 regulatory filing with the SEC.
> Apple Inc. (NASDAQ:AAPL) has risen 19.49% since May 1, 2017 and is uptrending.
> Longbow upgraded Apple Inc. (NASDAQ:AAPL) rating on Wednesday, March 30.

## 08 · AAPL · diagnostic_seen

**标题：** Hedge Funds Dump Most Apple Stock Since 2008: Full 13F Summary

- 记录：`2018_05_d157b48c57be246ec7dd80e7af4388a2.zip::news_0005518.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Hedge Funds Dump Most Apple Stock Since 2008: Full 13F Summary
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> According to Bloomberg calculations , in Q1 investors slashed their AAPL holdings by about 153 million shares : the biggest decrease since at least the first quarter of 2008.
> Yet while investor enthusiasm for Apple has somewhat lessened this year amid concern about whether the company will be able to sustain its pace of iPhone unit sales, though the stock is up 10 percent year-to-date, Tim Cook was saved by one simple thing: his stated intention to buybacks hundreds of billions in AAPL stocks which convinced the only person that matters to buy it.
> APPALOOSA MANAGEMENT Top new buys: LRCX, WFC, UBS, AMAT, SMH, AMLP, KNX, BYD, PAH, UAL Top exits: AAPL, EEM, CMCSA, MHK, VST, CSX, LUV Boosted stakes in MU, MGM, AGN, CNC, GOOG, LNG, DG, OC, PCG Cut stakes in QQQ, XLF, BAC, URI, NRG, WPZ, ETP, ALL, HCA

## 09 · AMZN · diagnostic_seen

**标题：** Facebook co-founder raises $40 million for streaming TV service Philo, launches service on Amazon Fire TV and Apple TV

- 记录：`2018_07_d157b48c57be246ec7dd80e7af4388a2.zip::news_0000379.json`
- 助手预先允许的事件：`product_business`
- 简单规则：`product_business`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Facebook co-founder raises $40 million for streaming TV service Philo, launches service on Amazon Fire TV and Apple TV
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> – Reporter, San Francisco Business Times Jul 11, 2018, 9:44am EDT San Francisco-based Philo is making big moves in the streaming space with $40 million in new funding and launches on Amazon Fire TV and Apple TV.
> Making Philo available on Amazon Fire TV (NASDAQ: AMZN) and Apple TV (NASDAQ: AAPL) was a top priority following the November launch because a number of people said they wanted to use Philo on more devices, Philo CEO Andrew McCollum told the San Francisco Business Times.

## 10 · AMZN · diagnostic_seen

**标题：** Tradewinds Capital Management Has Lifted Amazon Com (AMZN) Holding; Shorts at CHESSWOOD GROUP LTD ORDINARY SHARES CAN (CHWWF) Lowered By 32.7%

- 记录：`2018_06_d157b48c57be246ec7dd80e7af4388a2.zip::news_0004610.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S1`：Tradewinds Capital Management Llc increased Amazon Com Inc (AMZN) stake by 30.1% reported in 2018Q1 SEC filing.
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Tradewinds Capital Management Llc increased Amazon Com Inc (AMZN) stake by 30.1% reported in 2018Q1 SEC filing.
> Amazon.com, Inc. (NASDAQ:AMZN) has risen 62.09% outperformed by 49.52% AMZN News: ; 24/04/2018 – It took Jeff Bezos just 3 words to change the way Suzy Welch thinks about work.
> More notable recent Amazon.com, Inc. (NASDAQ:AMZN) news were published by: Seekingalpha.com which released: “Amazon’s New Secret To Accelerating Earnings Growth” on June 18, 2018, also Seekingalpha.com with their article: “Amazon launching Alexa for Hospitality in Marriott hotels” published on June 19, 2018, Fool.com published: “The Only Reason Jeff Bezos Would Let Amazon Split Its Stock” on June 19, 2018.

## 11 · AAPL · diagnostic_seen

**标题：** Baldwin Brothers Inc. MA Trims Holdings in Apple Inc. (AAPL)

- 记录：`2018_01_d157b48c57be246ec7dd80e7af4388a2.zip::news_0002292.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Baldwin Brothers Inc. MA Trims Holdings in Apple Inc. (AAPL)
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Baldwin Brothers Inc. MA Trims Holdings in Apple Inc. (AAPL) Posted by Alanna Baker | Jan 12th, 2018
> Baldwin Brothers Inc. MA cut its holdings in shares of Apple Inc. (NASDAQ:AAPL) by 4.7% in the third quarter, according to its most recent filing with the Securities and Exchange Commission.
> Apple (NASDAQ:AAPL) last announced its earnings results on Thursday, November 2nd.

## 12 · AMZN · diagnostic_seen

**标题：** Amazon sells more than 100 mln products on Prime Day event

- 记录：`2018_07_d157b48c57be246ec7dd80e7af4388a2.zip::news_0001854.json`
- 助手预先允许的事件：`product_business`
- 简单规则：`product_business`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Amazon sells more than 100 mln products on Prime Day event
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> July 18, 2018 / 1:48 PM / Updated an hour ago Speakers, TVs, Kleenex in demand on Amazon Prime Day Vibhuti Sharma 3 Min Read
> (Reuters) - Online shoppers purchased more than 100 million products worldwide during Amazon.com Inc’s ( AMZN.O ) annual Prime Day sale this week, despite glitches on its mobile app and websites that prevented several customers from placing orders.
> Amazon said its sales topped those for the Prime event a year ago as well as those for Cyber Monday and Black Friday, but it gave no breakdown of the value of sales or of the scale of discounts it had applied.

## 13 · AAPL · diagnostic_seen

**标题：** Stock Market Today: Nasdaq Up; Auto Chip Stock Breaks Out; Why Apple Is Still In A Buy Zone

- 记录：`2018_01_d157b48c57be246ec7dd80e7af4388a2.zip::news_0004645.json`
- 助手预先允许的事件：`market_recap`
- 简单规则：`other`
- 本地模型：`product_business`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S1`：Apple ( AAPL ), meanwhile, continues to act like a market leader ever since it turned fortunes around for investors with its breakout on Jan.
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple ( AAPL ), meanwhile, continues to act like a market leader ever since it turned fortunes around for investors with its breakout on Jan.
> Apple has not triggered any defense-type sell signals, such as a severe drop below the 50-day line or 10-week moving average in massive volume and failure to recover.
> Apple is expected to grow profits in the double digits for a fourth straight quarter as analysts polled by Thomson Reuters see earnings up 13% to $3.78 a share in the fiscal first quarter ended in December.

## 14 · AMZN · diagnostic_seen

**标题：** Kynikos Associates LP Has Lifted Holding in Western Digital (WDC) as Share Price Rose; Amazon.Com (AMZN) Shareholder Pcj Investment Counsel LTD Has Lifted Position by $455,910

- 记录：`2018_05_d157b48c57be246ec7dd80e7af4388a2.zip::news_0004027.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`product_business`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S1`：Pcj Investment Counsel Ltd increased its stake in Amazon.Com Inc (AMZN) by 46.43% based on its latest 2017Q4 regulatory filing with the SEC.
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Pcj Investment Counsel Ltd increased its stake in Amazon.Com Inc (AMZN) by 46.43% based on its latest 2017Q4 regulatory filing with the SEC.
> Amazon.com, Inc. (NASDAQ:AMZN) has risen 76.78% since May 10, 2017 and is uptrending.
> More notable recent Amazon.com, Inc. (NASDAQ:AMZN) news were published by: Nasdaq.com which released: “Amazon.com, Inc. (AMZN) Will Now Deliver to Your Car Trunk with Key In-Car” on April 25, 2018, also Investorplace.com with their article: “Amazon Just Went Into Beast Mode, But Don’t Buy Here” published on April 27, 2018, Nasdaq.com published: “Google to Acquire Velostrata in Effort to Make Enterprise Cloud Migration Easier” on May 10, 2018.

## 15 · AAPL · diagnostic_seen

**标题：** Vantage Investment Partners Has Cut By $14.98 Million Its Apple (AAPL) Position; Shorts at VEOLIA ENVIRONNEMENT ORDINARY SHARES FR (VEOEF) Raised By 29.93%

- 记录：`2018_04_d157b48c57be246ec7dd80e7af4388a2.zip::news_0002636.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Vantage Investment Partners Has Cut By $14.98 Million Its Apple (AAPL) Position; Shorts at VEOLIA ENVIRONNEMENT ORDINARY SHARES FR (VEOEF) Raised By 29.93%
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple Inc. (NASDAQ:AAPL) has risen 19.49%   uptrending.
> Analysts await Apple Inc. (NASDAQ:AAPL) to report earnings on May, 1 after the close.
> AAPL’s profit will be $13.75B for 16.04 P/E if the $2.71 EPS becomes a reality.

## 16 · AMZN · diagnostic_seen

**标题：** Amazon on track to cross $900 billion market-cap threshold - MarketWatch

- 记录：`2018_07_d157b48c57be246ec7dd80e7af4388a2.zip::news_0003033.json`
- 助手预先允许的事件：`market_recap`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Amazon on track to cross $900 billion market-cap threshold - MarketWatch
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Shares of Amazon.com Inc. AMZN, -2.98% shot up 4.3% toward a record high in premarket trade Friday, after blowout second-quarter results , to put the e-commerce giant on track to cross above the $900 billion market-capitalization threshold for first time.
> At current stock price levels, Amazon would be valued at about $919.4 billion.

## 17 · AAPL · diagnostic_seen

**标题：** Warren Buffett would ‘love to own 100%’ of AAPL as stock price set to hit new all-time high at market open

- 记录：`2018_05_d157b48c57be246ec7dd80e7af4388a2.zip::news_0002739.json`
- 助手预先允许的事件：`holdings|market_recap`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Warren Buffett would ‘love to own 100%’ of AAPL as stock price set to hit new all-time high at market open
- 诊断说明：Investor ownership preference and price milestone both present.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple shook off bearish analysts last week with solid earnings, guidance, and a new capital return program, with AAPL stock soaring after the results .
> In an interview with CNBC today, Buffet heaped even more positivity onto Apple, and the stock market is looking to propel AAPL even higher at the open.
> AAPL needs to breach $194 to hit that magic $1,000,000,000,000 number.

## 18 · AAPL · diagnostic_seen

**标题：** Amazon unveils Alexa feature before Apple releases HomePod

- 记录：`2018_01_d157b48c57be246ec7dd80e7af4388a2.zip::news_0005001.json`
- 助手预先允许的事件：`product_business`
- 简单规则：`product_business`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `none`：无
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Amazon unveils Alexa feature before Apple releases HomePod
> This comes days before Apple ( AAPL ) rolls out its own smart speaker, HomePod.
> Apple users will be able to send i-messages and texts using Siri.

## 19 · AAPL · diagnostic_seen

**标题：** As Apple (AAPL) Stock Value Rose, Biondo Investment Advisors Decreased by $1.06 Million Its Holding; First National Bank Of Mount Dora Trust Investment Services Lifted Eastman Chem Co (EMN) Holding by $715,050 as Share Price Rose

- 记录：`2018_06_d157b48c57be246ec7dd80e7af4388a2.zip::news_0003503.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：As Apple (AAPL) Stock Value Rose, Biondo Investment Advisors Decreased by $1.06 Million Its Holding; First National Bank Of Mount Dora Trust Investment Services Lifted Eastman Chem Co (EMN) Holding by $715,050 as Share Price Rose
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple Inc. (NASDAQ:AAPL) has risen 22.56% uptrending.
> Apple Inc. had 421 analyst reports since July 21, The stock of Apple Inc. (NASDAQ:AAPL) earned “Buy” rating by Nomura on Wednesday, August 2.
> AAPL’s profit will be $10.76B for 21.94 P/E if the $2.19 EPS becomes a reality.

## 40 · AAPL · new_check

**标题：** Apple has acquired Vancouver-based software development tools firm, Buddybuild

- 记录：`2018_01_d157b48c57be246ec7dd80e7af4388a2.zip::news_0000627.json`
- 助手预先允许的事件：`product_business`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Apple has acquired Vancouver-based software development tools firm, Buddybuild
- 诊断说明：Acquisition is commercial development.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Download PDF version 08:04 03 Jan 2018 According to the Canadian firm’s website, Buddybuild has joined the Xcode engineering group at Apple “to build amazing developer tools for the entire iOS community” Buddybuild - previously owned by Doe Pics Hit Inc. – added that it will continue to be based in Vancouver
> Apple Inc. ( NASDAQ:AAPL ) has acquired Buddybuild, a Vancouver based company that makes software development tools, according to the Canadian firm’s website .
> Apple has a habit of snapping up tech firms. At the end of November, technology news website techcrunch reported that the US tech giant had acquired Canadian start-up Vrvana, the maker of an augmented reality headset called Totem, for about US$30mln.

## 41 · AMZN · new_check

**标题：** As Boeing Co (BA) Valuation Declined, Beech Hill Advisors Decreased by $598,737 Its Stake; As Amazon (AMZN) Market Value Rose, Vantage Investment Advisors Has Increased Its Stake by $2.14 Million

- 记录：`2018_06_d157b48c57be246ec7dd80e7af4388a2.zip::news_0003210.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：As Boeing Co (BA) Valuation Declined, Beech Hill Advisors Decreased by $598,737 Its Stake; As Amazon (AMZN) Market Value Rose, Vantage Investment Advisors Has Increased Its Stake by $2.14 Million
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Amazon.com, Inc. (NASDAQ:AMZN) has risen 62.09% since June 12, 2017 and is uptrending.
> Raymond James upgraded Amazon.com, Inc. (NASDAQ:AMZN) rating on Thursday, August 27.
> The stock of Amazon.com, Inc. (NASDAQ:AMZN) has “Overweight” rating given on Tuesday, December 8 by Pacific Crest.

## 42 · AAPL · new_check

**标题：** Facebook, Apple, Amazon, Netflix, Google: Who Takes the Tech Crown in 2018?

- 记录：`2018_01_d157b48c57be246ec7dd80e7af4388a2.zip::news_0002057.json`
- 助手预先允许的事件：`market_recap`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `none`：无
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Of course, I’m referring to Facebook ( FB ), Apple ( AAPL ), Amazon ( AMZN ), Netflix ( NFLX ) and Google parent Alphabet ( GOOG , GOOGL ).
> The competitive markets, not to mention the sustained growth in mobile devices, should continue to drive Apple, Alphabet, Amazon and Microsoft higher.

## 43 · AMZN · new_check

**标题：** Technology Sector Update for 02/12/2018: MSFT, AAPL, IBM, CSCO, GOOG, AMZN, RPD, CRNT, FB

- 记录：`2018_02_d157b48c57be246ec7dd80e7af4388a2.zip::news_0001682.json`
- 助手预先允许的事件：`product_business`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Technology Sector Update for 02/12/2018: MSFT, AAPL, IBM, CSCO, GOOG, AMZN, RPD, CRNT, FB
- 诊断说明：Generic sector title; only target-specific body fact is acquisition.

**完整的本次目标摘录（不代表完整新闻正文）：**

> - Amazon ( AMZN ): reportedly paid $90 million to buy maker of Blink home security cameras

## 44 · AAPL · new_check

**标题：** Nextera Energy Partners LP (NEP) Holder Goldman Sachs Group Has Boosted Its Position; Eidelman Virant Capital Position in Apple (AAPL) Has Trimmed by $1.07 Million as Market Valuation Rose

- 记录：`2018_03_d157b48c57be246ec7dd80e7af4388a2.zip::news_0000861.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Nextera Energy Partners LP (NEP) Holder Goldman Sachs Group Has Boosted Its Position; Eidelman Virant Capital Position in Apple (AAPL) Has Trimmed by $1.07 Million as Market Valuation Rose
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple Inc. (NASDAQ:AAPL) has risen 62.48% since March 5, 2017 and is uptrending.
> Analysts await Apple Inc. (NASDAQ:AAPL) to report earnings on May, 1.
> AAPL’s profit will be $13.80B for 16.26 P/E if the $2.72 EPS becomes a reality.

## 45 · AMZN · new_check

**标题：** Facebook is in free fall and pulling the rest of tech with it (FB, AAPL, AMZN, NFLX, TWTR, GOOGL)

- 记录：`2018_07_d157b48c57be246ec7dd80e7af4388a2.zip::news_0003528.json`
- 助手预先允许的事件：`market_recap`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Facebook is in free fall and pulling the rest of tech with it (FB, AAPL, AMZN, NFLX, TWTR, GOOGL)
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Its tech peers — Amazon, Apple, Alphabet, and Netflix — also fell.
> Facebook : -24% Apple : -0.2% Amazon : -1% Netflix : -1% Alphabet (Google) : -0.6% Twitter , which is set to report earnings ahead of the opening bell on Friday, was down about 3% in early trading, fueled by criticism from President Donald Trump who accused the company of "shadow banning" prominent Republicans.
> Amazon is the only company of the tech basket left to report this week, and will release its quarterly results after the closing bell on Thursday.

## 46 · AAPL · new_check

**标题：** As Blackstone Group LP (BX) Share Value Declined, Randolph Co Increased Holding; Apple (AAPL) Market Valuation Rose While Ar Asset Management Has Cut Its Stake

- 记录：`2018_06_d157b48c57be246ec7dd80e7af4388a2.zip::news_0001406.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：As Blackstone Group LP (BX) Share Value Declined, Randolph Co Increased Holding; Apple (AAPL) Market Valuation Rose While Ar Asset Management Has Cut Its Stake
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple Inc. (NASDAQ:AAPL) has risen 22.56% outperformed by 9.99% the S&P500.
> Analysts await Apple Inc. (NASDAQ:AAPL) to report earnings on August, 7.
> AAPL’s profit will be $10.76 billion for 21.90 P/E if the $2.19 EPS becomes a reality.

## 47 · AMZN · new_check

**标题：** Amazon, Apple And Microsoft Bolster Consumer Discretionary And Tech Sector ETFs

- 记录：`2018_03_d157b48c57be246ec7dd80e7af4388a2.zip::news_0000136.json`
- 助手预先允许的事件：`market_recap`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Amazon, Apple And Microsoft Bolster Consumer Discretionary And Tech Sector ETFs
- 诊断说明：ETF/sector performance headline; weighting alone does not demonstrate change.

**完整的本次目标摘录（不代表完整新闻正文）：**

> The best performing ETF, Consumer Discretionary Select Sector SPDR, led with a 20% weighting in Amazon.com (AMZN).

## 48 · AAPL · new_check

**标题：** Apple Inc. (AAPL) Reaches $164.68 After 8.00% Up Move; Pandora Media (P) SI Increased By 8.33%

- 记录：`2018_04_d157b48c57be246ec7dd80e7af4388a2.zip::news_0003958.json`
- 助手预先允许的事件：`market_recap`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `none`：无
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple Inc. (AAPL) Reaches $164.68 After 8.00% Up Move; Pandora Media (P) SI Increased By 8.33%
> Apple Inc. (NASDAQ:AAPL) has risen 19.49% 7.94% the S&P500.The move comes after 5 months positive chart setup for the $835.59 billion company.
> AAPL’s profit will be $13.75 billion for 15.19 P/E if the $2.71 EPS becomes a reality.

## 49 · AMZN · new_check

**标题：** As General Mls (GIS) Valuation Declined, Shareholder Schnieders Capital Management Has Cut Its Holding; Amazon.Com (AMZN) Shareholder Huntington Steele Has Trimmed Its Position

- 记录：`2018_06_d157b48c57be246ec7dd80e7af4388a2.zip::news_0004447.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：As General Mls (GIS) Valuation Declined, Shareholder Schnieders Capital Management Has Cut Its Holding; Amazon.Com (AMZN) Shareholder Huntington Steele Has Trimmed Its Position
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Amazon.com, Inc. (NASDAQ:AMZN) has risen 62.09% since June 20, 2017 and is uptrending.
> Analysts await Amazon.com, Inc. (NASDAQ:AMZN) to report earnings on July, 26.
> AMZN’s profit will be $1.22 billion for 172.10 P/E if the $2.52 EPS becomes a reality.

## 50 · AAPL · new_check

**标题：** Apple reportedly cuts production of HomePod amid poor sales (AAPL) - plugilonew

- 记录：`2018_04_d157b48c57be246ec7dd80e7af4388a2.zip::news_0002846.json`
- 助手预先允许的事件：`product_business`
- 简单规则：`product_business`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S2`：Apple reportedly slashed production of its HomePod smart speaker and lowered sales forecasts for the device in late March, amid its ongoing struggles … 
- 诊断说明：Reported product production/sales outlook, not explicit management financial guidance.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Home Apple reportedly cuts production of HomePod amid poor sales (AAPL)
> Apple reportedly slashed production of its HomePod smart speaker and lowered sales forecasts for the device in late March, amid its ongoing struggles … 

## 51 · AAPL · new_check

**标题：** Sonos used a popular meme to troll Apple on HomePod launch day (AAPL)

- 记录：`2018_02_d157b48c57be246ec7dd80e7af4388a2.zip::news_0001757.json`
- 助手预先允许的事件：`product_business`
- 简单规则：`product_business`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S1`：Sonos used a popular meme to troll Apple on HomePod launch day (AAPL) By Kif Leswing Feb 9, 2018, 10:55 am Apple's HomePod hits stores on Friday, but Sonos, which makes competing smart speakers, isn't sweating it.
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Sonos used a popular meme to troll Apple on HomePod launch day (AAPL) By Kif Leswing Feb 9, 2018, 10:55 am Apple's HomePod hits stores on Friday, but Sonos, which makes competing smart speakers, isn't sweating it.

## 52 · AMZN · new_check

**标题：** Amazon Gains After Trouncing Earnings Estimates: 7 Key Takeaways - TheStreet

- 记录：`2018_07_d157b48c57be246ec7dd80e7af4388a2.zip::news_0003343.json`
- 助手预先允许的事件：`earnings|market_recap`
- 简单规则：`earnings`
- 本地模型：`earnings`
- 事件判断一致：True；事件与证据共同符合预先标记：True
- 模型所选证据 `S0`：Amazon Gains After Trouncing Earnings Estimates: 7 Key Takeaways - TheStreet
- 诊断说明：Earnings analysis and price reaction both central.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Amazon.com's ( AMZN ) margins continues to surge, and -- for now -- its trademark heavy spending growth is being dialed back a bit.
> Amazon Web Services (AWS) Remains a Juggernaut AWS revenue rose 49% for the second quarter in a row and totaled $6.11 billion, topping a $5.98 billion consensus.
> Amazon.com, Alphabet, Facebook and Microsoft are holdings in Jim Cramer's Action Alerts PLUS member club .

## 53 · AAPL · new_check

**标题：** Apple's stock set to trade below 200-day moving average first time since July 2016

- 记录：`2018_02_d157b48c57be246ec7dd80e7af4388a2.zip::news_0000295.json`
- 助手预先允许的事件：`market_recap`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Apple's stock set to trade below 200-day moving average first time since July 2016
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Shares of Apple Inc. (AAPL) fell 0.9% in premarket trade, putting it on track to trade below its 200-day moving average for the first time since July 28, 2016.
> Apple's stock has lost 7.0% over the past three months, while the Dow Jones Industrial Average has gained 8.4%.

## 54 · AAPL · new_check

**标题：** Mcmillion Capital Management Inc. Lowers Stake in Apple (AAPL)

- 记录：`2018_06_d157b48c57be246ec7dd80e7af4388a2.zip::news_0001910.json`
- 助手预先允许的事件：`holdings`
- 简单规则：`holdings`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Mcmillion Capital Management Inc. Lowers Stake in Apple (AAPL)
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Mcmillion Capital Management Inc. trimmed its holdings in shares of Apple (NASDAQ:AAPL) by 1.8% in the fourth quarter, according to the company in its most recent filing with the Securities and Exchange Commission.
> Shares of NASDAQ:AAPL opened at $193.31 on Wednesday.
> Apple (NASDAQ:AAPL) last issued its quarterly earnings results on Tuesday, May 1st.

## 55 · AMZN · new_check

**标题：** Why I Like Amazon, Apple and These Other Stocks for February

- 记录：`2018_02_d157b48c57be246ec7dd80e7af4388a2.zip::news_0000765.json`
- 助手预先允许的事件：`market_recap|holdings`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S1`：I own Amazon ( AMZN ) , Apple ( AAPL ) and Alphabet/Google ( GOOG ) , ( GOOGL ) .
- 诊断说明：Investment opinion plus existing ownership; no ownership change established.

**完整的本次目标摘录（不代表完整新闻正文）：**

> I own Amazon ( AMZN ) , Apple ( AAPL ) and Alphabet/Google ( GOOG ) , ( GOOGL ) .

## 56 · AAPL · new_check

**标题：** Apple-Aktie: Richtungsweisender Kampf

- 记录：`2018_02_d157b48c57be246ec7dd80e7af4388a2.zip::news_0002336.json`
- 助手预先允许的事件：`unknown`
- 简单规则：`other`
- 本地模型：`product_business`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S1`：Apple stört sich an Emoji-Verwendung in Apps ► Artikel lesen iPhone X verpasst wegen technischer Probleme Anrufe ► Artikel lesen Apple's $1,000 iPhone X has a bug that is preventing users from answering calls ► Artikel lesen Record profit for Apple despite disappointing sales ► Artikel lesen 14:49 3 Stocks to Watch on Monday: Amazon.com, Inc. (AMZN), Apple Inc. (AAPL) and Tesla Inc (TSLA) ► Artikel lesen
- 诊断说明：German headline plus unrelated linked headlines; insufficient coherent target event.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple stört sich an Emoji-Verwendung in Apps ► Artikel lesen iPhone X verpasst wegen technischer Probleme Anrufe ► Artikel lesen Apple's $1,000 iPhone X has a bug that is preventing users from answering calls ► Artikel lesen Record profit for Apple despite disappointing sales ► Artikel lesen 14:49 3 Stocks to Watch on Monday: Amazon.com, Inc. (AMZN), Apple Inc. (AAPL) and Tesla Inc (TSLA) ► Artikel lesen

## 57 · AMZN · new_check

**标题：** Here's what options traders expect from Apple, Amazon and Alphabet earnings

- 记录：`2018_02_d157b48c57be246ec7dd80e7af4388a2.zip::news_0000834.json`
- 助手预先允许的事件：`earnings|market_recap`
- 简单规则：`earnings`
- 本地模型：`earnings`
- 事件判断一致：True；事件与证据共同符合预先标记：True
- 模型所选证据 `S0`：Here's what options traders expect from Apple, Amazon and Alphabet earnings
- 诊断说明：Options expectations around earnings; both categories plausible.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Option straddles point to smaller-than-average one-day post-earnings moves for Apple and Amazon
> Investors shouldn't expect any extraordinary stock moves after Apple Inc., Amazon.com Inc. and Google parent Alphabet Inc. report earnings late Thursday, as options oddsmakers are expecting price reactions that are mostly below the longer-term averages.
> For Amazon (AMZN), the average move was 7.5% over the same period, with the 11 up days averaging a 7.7% gain and the 9 down days averaging a 7.2% decline.

## 58 · AAPL · new_check

**标题：** Apple is working on a cheap iPhone — and it feels like the iPhone 5C fiasco all over again (AAPL)

- 记录：`2018_01_d157b48c57be246ec7dd80e7af4388a2.zip::news_0003783.json`
- 助手预先允许的事件：`product_business`
- 简单规则：`product_business`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S2`：Apple customers who like the iPhone X’s facial recognition and edge-to-edge screen but were turned off by the $999 price tag may have additional options at lower prices this fall.
- 诊断说明：Primary target event from title and supplied excerpt; historical background excluded.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple is working on a cheap iPhone — and it feels like the iPhone 5C fiasco all over again (AAPL) Posted By: BusinessInsider January 29, 2018 Financial analysts predict Apple will launch three new iPhone models this year.
> Apple customers who like the iPhone X’s facial recognition and edge-to-edge screen but were turned off by the $999 price tag may have additional options at lower prices this fall.
> Apple is working on a lower-cost iPhone with some of the iPhone X’s best features for a launch later this year, according to KGI Securities analyst Ming-Chi Kuo.

## 59 · AAPL · new_check

**标题：** Apple (NASDAQ: AAPL ) has hired the tech team from Silicon V

- 记录：`2018_01_d157b48c57be246ec7dd80e7af4388a2.zip::news_0002620.json`
- 助手预先允许的事件：`other|product_business`
- 简单规则：`other`
- 本地模型：`earnings`
- 事件判断一致：False；事件与证据共同符合预先标记：False
- 模型所选证据 `S0`：Apple (NASDAQ: AAPL ) has hired the tech team from Silicon V
- 诊断说明：Team hire can be personnel or business development; no merger asserted.

**完整的本次目标摘录（不代表完整新闻正文）：**

> Apple (NASDAQ: AAPL ) has hired the tech team from Silicon Valley Data Science, a business transformation consulting startup, according to TechCrunch .
> Apple reportedly hired at least 18 members of the staff (roughly half to one-third of the total) including a co-founder/CTO and the other co-founder/CEO.
> Previously: Apple, suppliers drop as key analyst lowers iPhone X lifetime sales forecast (Jan.
