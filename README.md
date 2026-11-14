# Equimem: Continuous Conserved-Memory Engine

A flashcard and knowledge-state engine modeled as a **conserved scalar mass field** over frozen unit concept embeddings.

Traditional spaced repetition systems (SM-2, Anki, FSRS) rely on artificial wall-clock time intervals and independent binary cards. In contrast, **Equimem** (*Equilibrium* + *Memory*) models human memory as a continuous manifold where concepts interact through semantic similarity, and forgetting occurs through **competitive interference** rather than elapsed time.

---

## 1. How It Works

### Core Principles

1. **Unit Hypersphere Concept Embeddings ($\mathbb{S}^{D-1}$)**
   - Each flashcard (Target Question + Reference Answer + Domain) is projected into a dense semantic vector space via `gemini-embedding-001` (3,072 dimensions) and normalized to unit length: $\|v_i\| = 1.0$.
   - Cards that share conceptual foundations sit close to each other on the hypersphere (high cosine similarity $v_i^\top v_j$).

2. **Strict Mass Conservation ($\sum_{i=1}^N m_i = N$)**
   - Every card carries a scalar mass $m_i \ge 0$. The average mass across the deck is always $1.0$.
   - **Forgetting is competitive**: When you practice a card and get it right, its mass increases ($+\Delta m$), and this positive mass ripples outward to close neighbors.
   - To keep total mass constant at $N$, the engine scales down the mass of unreviewed cards proportionally. Knowledge you do not touch gradually deflates as you reinforce other areas.

3. **Smooth Neighborhood Mastery & Log-Gradients**
   - Mastery is not binary. The smoothed mastery at any point $x$ is computed using a radial similarity kernel:
     $$K(x, v_i) = \exp\left(\frac{x^\top v_i - 1}{\tau}\right)$$
   - Density-normalized smoothed mastery:
     $$\bar{m}(x) = \frac{\sum_i m_i K(x, v_i)}{\sum_i K(x, v_i)}$$
   - The log-gradient $\nabla_x \log \bar{m}(x) = \frac{1}{\tau}(\mathbb{E}_w[v] - \mathbb{E}_p[v])$ measures how rapidly mastery changes along the manifold, separating the student's mastery from raw cluster density.

4. **Card Selection & Active Frontier**
   - The engine selects cards by finding **valleys** (regions with lowest smoothed mastery $\bar{m}$) and **boundaries** (the frontier between mastered and unmastered concepts).
   - **Cooldown Exclusion**: Cards reviewed within the last $k$ steps are temporarily excluded ($S_i = -\infty$) to prevent immediate, repetitive prompting on the same item.

5. **Zero-Token Free-Form Grading (`gemini-decision-1`)**
   - Students answer in free-form natural language.
   - The response is evaluated by Google's `gemini-decision-1` Decisions classification head against the ground-truth Q&A pair.
   - This returns exact posterior probabilities $P(\text{Incorrect}), P(\text{Partial}), P(\text{Mastered})$ with **zero generative token overhead** in ~150 ms.

---

## 2. Project Architecture

```
equimem/
├── equimem/
│   ├── __init__.py           # Package exports
│   ├── models.py             # Data structures (FieldConfig, Card, FieldSnapshot, GradeResult)
│   ├── template.py           # Structured Q&A formatting for dense embedding
│   ├── embedder.py           # GeminiEmbedder (3072 dims) with local NGram fallback
│   ├── grader.py             # DecisionGrader (gemini-decision-1 evaluation head)
│   ├── engine.py             # MassFieldEngine (conservation, kernel, selection, diagnostics)
│   ├── mutator.py            # CardMutator (in-place mutation & expansion via gemini-3.5-flash-lite)
│   ├── tutor.py              # AITutor orchestration class
│   ├── server.py             # FastAPI REST server
│   └── mcp_server.py         # Model Context Protocol (MCP) server for AI assistants
├── lab/
│   ├── deck_50.py            # 60-card curated concept deck across 5 domains
│   ├── student_responses.py  # Realistic student answers (mastered, partial, misconceptions)
│   ├── simulate_50_cards.py  # 40-round end-to-end simulation harness
│   └── demo_tutor.py         # Live single-session interactive demo
├── tests/
│   ├── test_engine.py        # Mass conservation, kernel ripple, and selection unit tests
│   ├── test_grader.py        # Decision grader unit tests
│   ├── test_mcp.py           # Model Context Protocol (MCP) tool tests
│   ├── test_mutator.py       # Leave-one-out diagnostics & in-place mutation tests
│   ├── test_server.py        # FastAPI endpoint integration tests
│   └── test_tutor.py         # Full tutor workflow tests
├── pyproject.toml            # Project build configuration
└── .env                      # API keys (GOOGLE_API_KEY / GEMINI_API_KEY)
```

---

## 3. Experimental Results (40-Round Live Simulation)

We evaluated the engine across 40 continuous rounds on a 60-card deck spanning 5 scientific domains:
1. **Quantum Mechanics & Info** (Cards 0–11)
2. **Linear Algebra & Decompositions** (Cards 12–23)
3. **Probability & Information Theory** (Cards 24–35)
4. **Algorithms & Complexity** (Cards 36–47)
5. **Deep Learning & Optimization** (Cards 48–59)

### Phase 1: Uneven Student Profile (Rounds 1–20)
- **Profile**: Student is strong in Linear Algebra & Algorithms (`y ≈ 1.0`), partial in Info Theory (`y ≈ 0.6`), and holds misconceptions in Quantum Mechanics (`y ≈ 0.0`).
- **Grader Output**:
  - Quantum error (e.g. No-Cloning theorem misconception): `P0 = 0.98, P1 = 0.02, P2 = 0.00` → `Score = 0.010`. Card mass dropped from `0.970` to `0.468`.
  - Algorithm success (e.g. Quickselect average time): `P0 = 0.00, P1 = 0.02, P2 = 0.98` → `Score = 0.992`. Card mass rose from `1.000` to `1.569`.
- **Selection Behavior**: The engine identified the negative gradient in Quantum Mechanics and concentrated review turns on the Quantum valley (Cards 04, 02, 08, 05, 09, 11, 01).

### Phase 2: Active Remediation & Dynamic Shift (Rounds 21–40)
- **Remediation**: At Round 21, the student studied Quantum Mechanics, answering with high accuracy (`y ≥ 0.94`).
- **Surface Adaptation**:
  - As Quantum card masses rose (`0.77` to `1.31`), the Quantum valley was eliminated (average QM mastery rose to `1.0238`).
  - Once the valley flattened, the engine **autonomously shifted focus** away from Quantum to sample other unmastered concepts: Linear Algebra (Spectral Theorem, QR decomposition), Information Theory (Gibbs sampling, Bayes), and Deep Learning (LoRA, Attention).

### Empirical Mastery Distribution

| Domain | Card Count | Initial Mastery m̄ (T=0) | Phase 1 Mastery m̄ (T=20) | Final Mastery m̄ (T=40) |
| :--- | :---: | :---: | :---: | :---: |
| **Algorithms** | 12 | 1.0000 | 1.0342 | 0.9749 |
| **Deep Learning** | 12 | 1.0000 | 0.9961 | 0.9912 |
| **Information Theory** | 12 | 1.0000 | 1.0118 | 1.0055 |
| **Linear Algebra** | 12 | 1.0000 | 0.9975 | 1.0230 |
| **Quantum Mechanics** | 12 | 1.0000 | **0.9604** (Valley) | **1.0238** (Remediated) |
| **Global Invariant (Total Mass)** | **60** | **60.000000** | **60.000000** | **60.000000** |
| **Surface Variance Var(m̄)** | — | 0.00000 | 0.00318 | **0.00054** |

Total scalar mass remained **strictly 60.000000** at every single update.

---

## 4. Latent Space Trajectory Search

The engine supports multi-branch gradient drift retrieval along the hypersphere:

```python
# Downhill search: follow negative gradient to find the student's biggest gaps around a topic
traj_down = engine.trajectory_search("quantum states and superposition", direction="downhill")

# Uphill search: follow positive gradient to find strong mastery anchors
traj_up = engine.trajectory_search("eigenvalues and matrix decomposition", direction="uphill")
```

- **Downhill** finds residual unmastered cards in that neighborhood.
- **Uphill** navigates along the gradient to identify the highest-mastery anchor concepts.

---

## 5. Quickstart & Usage

### 1. Prerequisites
Install dependencies and configure your Google Gemini API key:

```bash
pip install google-genai numpy pytest python-dotenv
```

Add your API key to `.env`:
```ini
GOOGLE_API_KEY=your_actual_api_key_here
```

### 2. Run Tests
Verify engine mechanics and mass conservation:

```bash
python -m pytest
```

### 3. Run the 40-Round Simulation
Run the full 60-card simulation across all 5 domains:

```bash
python lab/simulate_50_cards.py
```
*(Card embeddings are cached to `lab/cache_embeddings_60.npz` after the first run for instant execution.)*

### 4. Basic Code Example

```python
from equimem.models import FieldConfig
from equimem.engine import MassFieldEngine
from equimem.embedder import GeminiEmbedder
from equimem.grader import DecisionGrader

# 1. Initialize components
embedder = GeminiEmbedder(model="gemini-embedding-001")
grader = DecisionGrader(model="gemini-decision-1")
engine = MassFieldEngine(embedder=embedder, config=FieldConfig(tau=0.15))

# 2. Add concept cards
engine.add_card(
    question="What is Singular Value Decomposition (SVD)?",
    answer="Factorization A = U Sigma V-transpose into orthogonal and diagonal matrices.",
    topic="Linear Algebra"
)

# 3. Select next card to review (excluding recently seen cards)
card_id = engine.select_next(strategy="boundary", exclude=[])
card = engine.cards[card_id]

# 4. Grade free-form answer
student_answer = "Decomposition of a matrix into two orthogonal matrices and singular values."
result = grader.grade(card=card, student_answer=student_answer)

# 5. Update conserved mass field
engine.review(card_id, score=result.score)
print(f"Reviewed. Score: {result.score:.2f} | Current Deck Mass Sum: {engine.m.sum():.4f}")
```

---

## 6. FastAPI REST Server

Equimem includes a production-ready FastAPI server for any web UI, mobile app, or external agent service:

```bash
# Start the API server on http://localhost:8000
python -m equimem.server
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### Key REST Endpoints:
- `GET /api/health` — Returns deck size, total mass, and model connectivity status.
- `GET /api/state` — Full knowledge state snapshot (surface variance, entropy, per-topic metrics).
- `GET /api/cards` — Retrieve all concept cards with current mass and smoothed mastery.
- `POST /api/cards` — Register one or more new concept cards into the manifold.
- `POST /api/session/next` — Fetch the next recommended question (withholding answer).
- `POST /api/session/review` — Submit student response for zero-token evaluation and mass update.
- `POST /api/search/trajectory` — Run spherical gradient drift search (`downhill` gaps or `uphill` anchors).

---

## 7. Model Context Protocol (MCP) Server for AI Assistants

Equimem provides a native MCP server for integration with LLM coding agents and assistants (Claude Desktop, Cursor, Antigravity, etc.).

### Run Standalone:
```bash
python -m equimem.mcp_server
```

### Configuration (e.g. `claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "equimem": {
      "command": "python",
      "args": ["-m", "equimem.mcp_server"],
      "env": {
        "GOOGLE_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

### Exposed MCP Tools:
1. `select_next_card(strategy="boundary", temperature=0.10, cooldown=8)` — Selects the next card based on active learning boundary or lowest-mastery valleys.
2. `review_card(card_id, student_answer)` — Grades free-form recall with Gemini Decision 1 and dynamically updates the conserved mass field.
3. `add_card(question, answer, topic)` — Adds a new concept card to the conserved manifold.
4. `trajectory_search(query, direction="downhill", k=5)` — Discovers gaps or mastery anchors by following spherical manifold gradients.
5. `get_knowledge_state()` — Returns overall diagnostic metrics, surface variance, and domain distributions.
6. `diagnose_card(card_id)` — Performs Leave-One-Out Regional Diagnostics to detect spikes, plateaus, and potholes.
7. `evolve_card(card_id)` — On-demand in-place card replacement for isolated spikes or reframing for potholes.
8. `expand_cluster(card_id, count=2)` — On-demand outward addition of new frontier concept cards for consolidated plateaus.

---

## 8. On-Demand Card Evolution & Mutation

Standard spaced repetition treats cards in complete isolation. Equimem leverages the continuous semantic neighborhood to mathematically distinguish between rote phrasing memorization and genuine conceptual understanding:

### Mathematical Formulation: Leave-One-Out Neighborhood Mastery

For any card $v_i$, we extract its surrounding neighborhood mastery excluding card $i$ itself:

$$\bar{m}_{-i} = \frac{\Phi(v_i) - m_i}{\rho_0(v_i) - 1.0} = \frac{\sum_{j \neq i} m_j K(v_i, v_j)}{\sum_{j \neq i} K(v_i, v_j)}$$

The **Spike Residual** measures the discrepancy between local card score and neighborhood mastery:

$$\Delta_i = m_i - \bar{m}_{-i}$$

### Geometric Signatures and Actions

| Geometric Signature | Mathematical Condition | Cognitive Interpretation | Engine Action |
| :--- | :--- | :--- | :--- |
| **1. Isolated Spike** | $m_i \ge 1.25$, $\Delta_i \ge 0.30$ | The student aced this specific question, but adjacent semantic neighbors remain weak. The card is either too easy or phrased too simply. | **In-Place Mutation**: Upgrades card $i$ into a deeper, mechanism-level synthesis bridging to its weak neighbors. Mass relaxes to $\bar{m}_{-i}$, and conserved mass is redistributed. |
| **2. Consolidated Plateau** | $m_i \ge 1.15$, $\bar{m}_{-i} \ge 1.05$, $\|\Delta_i\| \le 0.35$ | Both the card and its entire semantic neighborhood are mastered. Conceptual ownership is robust across the cluster. | **Outward Expansion**: Generates brand-new concept cards at the frontier boundary of the cluster ($m = 0.0$). |
| **3. Isolated Pothole** | $m_i \le 0.75$, $\bar{m}_{-i} \ge 1.05$, $\Delta_i \le -0.30$ | The student understands the domain well, but repeatedly misses this specific card (bad phrasing or conceptual blind spot). | **Anchor Bridge Reframing**: Rewrites card $i$ anchored directly to the strong surrounding concepts already mastered. |
| **4. Developing Valley** | Otherwise | The region is still emerging or being consolidated. | **Normal Review**: Standard gradient-based review selection. |

> **Strictly Manual Execution**: Generative LLM mutation and expansion are never executed automatically during routine review loops. They are triggered on demand via explicit assistant tool calls (`evolve_card`, `expand_cluster`) or REST endpoints (`/api/cards/{id}/evolve`, `/api/cards/{id}/expand`).


