# MICA Test Queries

These queries are designed to test different aspects of the MICA system.

---

## Simple Queries (Should Skip Planning)

### Query 1: Tool Information
```
What tools are available in MICA?
```
**Expected**: Direct response listing the 6 available tools without creating a plan.

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

## Expected Outputs

- **Simple queries**: Direct text response in 5-10 seconds
- **Complex queries**:
  - Plan generation: 15-30 seconds
  - Execution: 1-5 minutes depending on steps
  - PDF report generated in session folder
