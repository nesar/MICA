# MICA Test Queries

These queries are designed to test different aspects of the MICA system.

---

## Simple Queries (Should Skip Planning)

### Query 1: Tool Information
```
What tools are available in MICA?
```
**Expected**: Direct response listing the 8 available tools without creating a plan.

### Query 2: Basic Definition
```
What is lithium?
```
**Expected**: Direct answer about lithium without multi-step analysis.

---

## Complex Queries (Require Full Planning)

### Query 3: Supply Chain Analysis
```
What are the top 3 critical materials in Greenland that are not available in mainland US or Alaska? Provide a timeline of developments from 2019-2024.
```
**Expected Workflow**:
1. Preliminary research phase
2. Plan generation with 8-12 steps including web searches
3. User approval required
4. Execution of search and analysis steps
5. Final summary and PDF report

**Key Topics to Cover**:
- Rare Earth Elements (REEs) - Kvanefjeld project
- Graphite deposits
- Uranium mining policy changes (2021 ban)
- US-Greenland cooperation agreements
- Chinese investment restrictions

### Query 4: Domestic Production Assessment
```
Analyze the current state of US domestic rare earth element production capacity. What percentage of US demand is met domestically vs imported? Include data from USGS and DOE reports.
```
**Expected Workflow**:
1. Web search for USGS Mineral Commodity Summaries
2. DOE critical materials reports
3. Trade data analysis
4. Production capacity assessment
5. Import dependency calculations
6. Final report with statistics

**Key Topics to Cover**:
- Mountain Pass mine (California)
- MP Materials production data
- Heavy vs light REE separation capacity
- China import dependency (~80%)
- DOE supply chain initiatives

### Query 5: Battery Materials Supply Chain
```
Compare lithium supply chains between the US and China. What are the key vulnerabilities in the US battery supply chain for electric vehicles?
```
**Expected Workflow**:
1. Research US lithium production (Nevada, California)
2. Research China's lithium dominance
3. Battery-grade processing capacity comparison
4. Supply chain mapping
5. Risk assessment
6. Policy recommendations

**Key Topics to Cover**:
- Lithium extraction methods (brine vs hard rock)
- Thacker Pass project
- China's Jiangxi province dominance
- Battery recycling potential
- IRA incentives impact

### Query 6: Policy Impact Analysis
```
How has the Inflation Reduction Act (IRA) affected critical mineral sourcing requirements for EV tax credits? What materials are most impacted?
```
**Expected Workflow**:
1. IRA provisions research
2. Critical mineral sourcing requirements
3. Free Trade Agreement (FTA) implications
4. Battery component requirements timeline
5. Industry response analysis
6. Compliance challenges

**Key Topics to Cover**:
- 40%/80% critical mineral thresholds
- FTA partner countries
- Battery component requirements
- OEM compliance strategies
- Treasury guidance updates

---

## Testing Instructions

### Test Simple Query:
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What tools are available?", "user_id": "test"}'
```
Should complete quickly without plan approval step.

### Test Complex Query:
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the top 3 critical materials in Greenland that are not available in mainland US or Alaska?", "user_id": "test"}'
```
Should generate a plan and wait for approval.

### Via Open WebUI:
1. Start backend: `cd backend && ARGO_USERNAME=your_user python -m mica.api.main`
2. Start pipelines: `cd ui && python pipelines_server.py`
3. Configure Open WebUI with `http://host.docker.internal:9099/v1`
4. Select "mica-analyst" model
5. Enter query and wait for plan
6. Type "approve" to execute

---

## Local Database Queries (Utilize Local PDFs and Data Files)

### Query 7: DOE Critical Materials Assessment
```
Using the DOE Critical Materials Assessment report in the local database, what are the top 5 most critical materials for clean energy technologies and what supply chain risks do they face?
```
**Expected Workflow**:
1. Search local PDF database for DOE critical materials report
2. Extract key findings on material criticality
3. Identify supply chain risk factors
4. Summarize mitigation strategies
5. Generate report with citations

**Tools Used**: `local_doc_search`, `code_agent`, `doc_generator`

### Query 8: USGS Production Data Analysis
```
Analyze the production trends for rare earth elements using the USGS data in the local database. Create visualizations showing year-over-year changes.
```
**Expected Workflow**:
1. List available data files in local database
2. Read USGS production data (Excel/CSV)
3. Perform statistical analysis on trends
4. Generate time series visualizations
5. Create summary report

**Tools Used**: `local_data_analysis`, `code_agent`, `doc_generator`

### Query 9: Federal Register Critical Minerals List
```
What materials are on the 2023 DOE Critical Materials List? How does this compare to the 2022 USGS Critical Minerals list? Search the Federal Register documents in the local database.
```
**Expected Workflow**:
1. Search local PDFs for Federal Register documents
2. Extract 2023 DOE Critical Materials List
3. Extract 2022 USGS Critical Minerals List
4. Compare and contrast the two lists
5. Identify additions and removals
6. Generate comparison report

**Tools Used**: `local_doc_search`, `code_agent`, `doc_generator`

### Query 10: Combined Local and Web Analysis
```
Using the DOE reports in the local database and current web sources, what progress has been made on domestic rare earth processing capacity since 2021?
```
**Expected Workflow**:
1. Search local PDFs for DOE baseline data
2. Web search for recent developments
3. Compare historical vs current capacity
4. Analyze investment and policy changes
5. Generate timeline and progress report

**Tools Used**: `local_doc_search`, `web_search`, `code_agent`, `doc_generator`

---

## Semiconductor and Advanced Materials Queries

### Query 11: Gallium and Germanium Supply Analysis
```
Analyze the supply chain implications of China's export controls on gallium and germanium. What are the US dependencies and potential alternatives?
```
**Expected Workflow**:
1. Web search for China export control details
2. Research US consumption of Ga and Ge
3. Identify current suppliers
4. Analyze potential alternative sources
5. Assess recycling potential
6. Generate risk assessment report

**Key Topics to Cover**:
- Semiconductor-grade gallium requirements
- Germanium in fiber optics and IR applications
- Byproduct production economics
- Japan, Canada, and European alternatives

### Query 12: Cobalt-Free Battery Analysis
```
What progress has been made on cobalt-free battery technologies? How would widespread adoption affect the critical materials supply chain?
```
**Expected Workflow**:
1. Research LFP and sodium-ion developments
2. Compare material requirements vs NMC/NCA
3. Analyze cost and performance tradeoffs
4. Assess supply chain implications
5. Generate technology comparison report

**Key Topics to Cover**:
- CATL and BYD LFP adoption
- Tesla's LFP shift for standard range
- Sodium-ion commercialization timeline
- Lithium demand impact scenarios

---

## Quantitative Analysis Queries

### Query 13: Import Dependency Calculations
```
Calculate the US import dependency for the following critical materials: rare earths, lithium, cobalt, graphite, and manganese. Use USGS trade data and present results with visualizations.
```
**Expected Workflow**:
1. Gather production data from local database
2. Web search for import/export statistics
3. Calculate net import reliance percentages
4. Create bar chart comparison
5. Identify primary source countries
6. Generate statistical summary

**Tools Used**: `local_data_analysis`, `web_search`, `code_agent`, `doc_generator`

### Query 14: Price Trend Analysis
```
Analyze price trends for lithium carbonate, cobalt, and nickel from 2020-2024. Identify key events that caused price spikes or crashes.
```
**Expected Workflow**:
1. Search for historical price data
2. Create time series visualizations
3. Annotate major price events
4. Calculate volatility metrics
5. Correlate with supply/demand events
6. Generate market analysis report

**Tools Used**: `web_search`, `local_data_analysis`, `code_agent`, `doc_generator`

---

## Policy and Regulatory Queries

### Query 15: Defense Production Act Usage
```
How has the Defense Production Act (DPA) been used to support critical mineral supply chains? List specific Title III projects and their funding amounts.
```
**Expected Workflow**:
1. Search for DPA Title III announcements
2. Compile project list with funding
3. Categorize by material type
4. Analyze geographic distribution
5. Generate project summary table

**Key Topics to Cover**:
- MP Materials DPA award
- Rare earth processing investments
- Battery materials projects
- DOD strategic priorities

### Query 16: International Agreements Analysis
```
What bilateral agreements has the US signed related to critical minerals? Analyze the Minerals Security Partnership and its member countries.
```
**Expected Workflow**:
1. Research Minerals Security Partnership
2. Identify member countries
3. Catalog bilateral critical mineral agreements
4. Analyze strategic objectives
5. Generate partnership assessment

**Key Topics to Cover**:
- MSP founding and objectives
- Australia, Canada, Japan agreements
- EU Critical Raw Materials Act alignment
- Friend-shoring strategies

---

## Environmental and Sustainability Queries

### Query 17: Rare Earth Recycling Potential
```
What is the current state of rare earth recycling from end-of-life products? Analyze recovery rates, economics, and scaling challenges.
```
**Expected Workflow**:
1. Research current recycling technologies
2. Analyze recovery rates by application
3. Assess economic viability
4. Identify scaling barriers
5. Review DOE recycling initiatives
6. Generate technology assessment

**Key Topics to Cover**:
- Magnet recycling from hard drives and EVs
- Urban mining potential
- Hydrometallurgical vs pyrometallurgical processes
- CMI and ReCell Center research

### Query 18: Mining Environmental Impact Assessment
```
Compare the environmental footprint of lithium extraction from brine operations (Atacama) vs hard rock mining (Australia). Include water usage and carbon intensity.
```
**Expected Workflow**:
1. Research brine extraction process
2. Research hard rock (spodumene) mining
3. Compare water consumption metrics
4. Analyze energy/carbon intensity
5. Assess ecosystem impacts
6. Generate comparative assessment

---

## Follow-up Query Testing

### Query 19: Initial Query with Follow-up
**First Query**:
```
What is the current US production capacity for lithium?
```
**Follow-up Query**:
```
What about projected capacity by 2030 with announced projects?
```
**Expected**: System should maintain context from first query and incorporate projections.

### Query 20: Refinement Query
**First Query**:
```
Analyze critical materials for offshore wind turbines.
```
**Follow-up Query**:
```
Focus specifically on permanent magnet materials and compare direct-drive vs geared turbine requirements.
```
**Expected**: System should narrow analysis based on feedback while retaining original context.

---

## Expected Outputs

- **Simple queries**: Direct text response in 5-10 seconds
- **Complex queries**:
  - Plan generation: 15-30 seconds
  - Execution: 1-5 minutes depending on steps
  - PDF report generated in session folder
- **Local database queries**:
  - Faster than web-only queries
  - Higher accuracy from curated sources
  - Charts and visualizations from local data files
- **Combined queries**:
  - Blend of local expertise and current web data
  - Best for tracking developments against baseline reports
