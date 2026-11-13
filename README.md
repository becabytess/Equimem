# Conserved-Mass Geometric Anki Engine

A flashcard and knowledge-state engine modeled as a **conserved scalar mass field** over frozen unit concept embeddings.

Traditional spaced repetition systems (SM-2, Anki, FSRS) rely on artificial wall-clock time intervals and independent binary cards. In contrast, this engine models human memory as a continuous manifold where concepts interact through semantic similarity, and forgetting occurs through **competitive interference** rather than elapsed time.

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
geo_anki/
├── geo_anki/
│   ├── __init__.py           # Package exports
│   ├── models.py             # Data structures (FieldConfig, Card, FieldSnapshot, GradeResult)
│   ├── template.py           # Structured Q&A formatting for dense embedding
│   ├── embedder.py           # GeminiEmbedder (3072 dims) with local NGram fallback
│   ├── grader.py             # DecisionGrader (gemini-decision-1 evaluation head)
│   ├── engine.py             # MassFieldEngine (conservation, kernel, selection, trajectories)
│   └── tutor.py              # AITutor orchestration class
├── lab/
│   ├── deck_50.py            # 60-card curated concept deck across 5 domains
│   ├── student_responses.py  # Realistic student answers (mastered, partial, misconceptions)
│   ├── simulate_50_cards.py  # 40-round end-to-end simulation harness
│   └── demo_tutor.py         # Live single-session interactive demo
├── tests/
│   ├── test_engine.py        # Mass conservation, kernel ripple, and selection unit tests
│   ├── test_grader.py        # Decision grader unit tests
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
- **Profile**: Student is strong in Linear Algebra & Algorithms ($y \approx 1.0$), partial in Info Theory ($y \approx 0.6$), and holds misconceptions in Quantum Mechanics ($y \approx 0.0$).
- **Grader Output**:
  - Quantum error (e.g. No-Cloning theorem misconception): $P_0 = 0.98, P_1 = 0.02, P_2 = 0.00 \implies \text{Score} = 0.010$. Card mass dropped from $0.970 \to 0.468$.
  - Algorithm success (e.g. Quickselect average time): $P_0 = 0.00, P_1 = 0.02, P_2 = 0.98 \implies \text{Score} = 0.992$. Card mass rose from $1.000 \to 1.569$.
- **Selection Behavior**: The engine identified the negative gradient in Quantum Mechanics and concentrated review turns on the Quantum valley (Cards 04, 02, 08, 05, 09, 11, 01).

### Phase 2: Active Remediation & Dynamic Shift (Rounds 21–40)
- **Remediation**: At Round 21, the student studied Quantum Mechanics, answering with high accuracy ($y \ge 0.94$).
- **Surface Adaptation**:
  - As Quantum card masses rose ($0.77 \to 1.31$), the Quantum valley was eliminated ($\bar{m}_{QM}$ rose to $1.0238$).
  - Once the valley flattened, the engine **autonomously shifted focus** away from Quantum to sample other unmastered concepts: Linear Algebra (Spectral Theorem, QR decomposition), Information Theory (Gibbs sampling, Bayes), and Deep Learning (LoRA, Attention).

### Empirical Mastery Distribution

| Domain | Card Count | Initial $\bar{m}$ (T=0) | Phase 1 $\bar{m}$ (T=20) | Final $\bar{m}$ (T=40) |
| :--- | :---: | :---: | :---: | :---: |
| **Algorithms** | 12 | 1.0000 | 1.0342 | 0.9749 |
| **Deep Learning** | 12 | 1.0000 | 0.9961 | 0.9912 |
| **Information Theory** | 12 | 1.0000 | 1.0118 | 1.0055 |
| **Linear Algebra** | 12 | 1.0000 | 0.9975 | 1.0230 |
| **Quantum Mechanics** | 12 | 1.0000 | **0.9604** (Valley) | **1.0238** (Remediated) |
| **Global Invariant ($\sum m_i$)** | **60** | **60.000000** | **60.000000** | **60.000000** |
| **Surface Variance Var(m_bar)** | — | 0.00000 | 0.00318 | **0.00054** |

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

Add your API key to `geo_anki/.env`:
```ini
GOOGLE_API_KEY=your_actual_api_key_here
```

### 2. Run Tests
Verify engine mechanics and mass conservation:

```bash
cd geo_anki
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
from geo_anki.models import FieldConfig
from geo_anki.engine import MassFieldEngine
from geo_anki.embedder import GeminiEmbedder
from geo_anki.grader import DecisionGrader

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
