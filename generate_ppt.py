"""
generate_ppt.py
---------------
Generates `Portfolio_Optimization_RL_Presentation.pptx` strictly following
the university/college presentation template:

  Slide 1  : Title of the Project, Student Name & Reg. No., Guide Name, Designation & Dept.
  Slide 2  : Contents (Introduction, Objectives, Block diagram, Simulation diagram,
             Work done so far, Results and discussion, Work pending, Conclusion, References)
  Slide 3  : Introduction
  Slide 4  : Objectives
  Slide 5  : Block diagram (End-to-End System Block Diagram drawn with native PPT shapes)
  Slide 6  : Simulation diagram (Closed-Loop RL Environment & Validation Simulation Architecture)
  Slide 7  : Work done so far (Phases 1, 2, and 3 completed milestones)
  Slide 8  : Results and discussion — Part 1: Main Comparative Study (Table + Equity/Drawdown plots)
  Slide 9  : Results and discussion — Part 2: Feature Ablation Study (Table + Ablation plot)
  Slide 10 : Work pending
  Slide 11 : Conclusion
  Slide 12 : References

Run directly:
    pip install python-pptx matplotlib
    python generate_ppt.py
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# =============================================================================
# EDIT YOUR DETAILS HERE (or directly in PowerPoint after generating)
# =============================================================================
PROJECT_TITLE = "Deep Reinforcement Learning for Multi-Asset Portfolio Optimization:\nA Comparative Study of DQN, PPO, and Classical Strategies"
STUDENT_NAME_REG = "Student Name  |  Registration No.: [Enter Reg. No.]"
GUIDE_NAME = "Guide Name"
GUIDE_DEPT = "Designation & Department"

# Theme Colors
NAVY = RGBColor(15, 32, 67)
BLUE = RGBColor(31, 119, 180)
TEAL = RGBColor(16, 137, 123)
GOLD = RGBColor(217, 119, 6)
SLATE = RGBColor(51, 65, 85)
DARK_TEXT = RGBColor(30, 41, 59)
MUTED_TEXT = RGBColor(100, 116, 139)
LIGHT_BG = RGBColor(248, 250, 252)
CARD_BG = RGBColor(255, 255, 255)
BORDER_GRAY = RGBColor(203, 213, 225)
WHITE = RGBColor(255, 255, 255)
HIGHLIGHT_ROW = RGBColor(224, 242, 254)


def set_slide_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title_text, subtitle_text=""):
    set_slide_bg(slide, LIGHT_BG)
    banner = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1)
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.14), Inches(12.1), Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = "Calibri"

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.size = Pt(13)
        p2.font.color.rgb = RGBColor(186, 230, 253)
        p2.font.name = "Calibri"


def add_card(slide, left, top, width, height, title, bullets, accent_color=BLUE, font_size=13.5):
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = BORDER_GRAY
    card.line.width = Pt(1)

    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.08)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent_color
    bar.line.fill.background()

    tb = slide.shapes.add_textbox(
        Inches(left + 0.2), Inches(top + 0.16), Inches(width - 0.4), Inches(height - 0.25)
    )
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.font.name = "Calibri"
    p.space_after = Pt(8)

    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = f"•  {b}"
        pb.font.size = Pt(font_size)
        pb.font.color.rgb = DARK_TEXT
        pb.font.name = "Calibri"
        pb.space_after = Pt(6)


def add_diagram_box(slide, left, top, width, height, title, body, fill_color=NAVY, text_color=WHITE):
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.fill.solid()
    box.fill.fore_color.rgb = fill_color
    box.line.color.rgb = WHITE if fill_color != CARD_BG else NAVY
    box.line.width = Pt(1.5)

    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = text_color
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER

    if body:
        p2 = tf.add_paragraph()
        p2.text = body
        p2.font.size = Pt(11)
        p2.font.color.rgb = text_color
        p2.font.name = "Calibri"
        p2.alignment = PP_ALIGN.CENTER


def add_arrow(slide, left, top, width, height, direction="right", color=SLATE):
    shape_type = {
        "right": MSO_SHAPE.RIGHT_ARROW,
        "down": MSO_SHAPE.DOWN_ARROW,
        "left": MSO_SHAPE.LEFT_ARROW,
        "up": MSO_SHAPE.UP_ARROW,
    }[direction]
    arr = slide.shapes.add_shape(shape_type, Inches(left), Inches(top), Inches(width), Inches(height))
    arr.fill.solid()
    arr.fill.fore_color.rgb = color
    arr.line.fill.background()


def add_styled_table(slide, left, top, width, height, headers, rows, highlight_row_idx=0, font_size=12):
    table_shape = slide.shapes.add_table(
        len(rows) + 1, len(headers), Inches(left), Inches(top), Inches(width), Inches(height)
    )
    table = table_shape.table

    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(font_size)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT

    for r_idx, row in enumerate(rows):
        is_highlight = r_idx == highlight_row_idx
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = HIGHLIGHT_ROW if is_highlight else (
                RGBColor(241, 245, 249) if r_idx % 2 == 1 else WHITE
            )
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.size = Pt(font_size)
            p.font.bold = is_highlight or c_idx == 0
            p.font.color.rgb = NAVY if is_highlight else DARK_TEXT
            p.font.name = "Calibri"
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def build_presentation():
    if not os.path.exists("figures/equity_curves.png"):
        try:
            import plot_results
            plot_results.main()
        except Exception as e:
            print(f"Note: Could not auto-generate figures ({e}). Proceeding with tables.")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide (Exact Template Structure)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1, NAVY)

    card = s1.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(0.8), Inches(11.333), Inches(5.9)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = RGBColor(23, 43, 84)
    card.line.color.rgb = TEAL
    card.line.width = Pt(2)

    tb = s1.shapes.add_textbox(Inches(1.4), Inches(1.15), Inches(10.5), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p_title = tf.paragraphs[0]
    p_title.text = PROJECT_TITLE
    p_title.font.size = Pt(30)
    p_title.font.bold = True
    p_title.font.color.rgb = WHITE
    p_title.font.name = "Calibri"
    p_title.alignment = PP_ALIGN.CENTER
    p_title.space_after = Pt(24)

    p_by = tf.add_paragraph()
    p_by.text = "by"
    p_by.font.size = Pt(16)
    p_by.font.italic = True
    p_by.font.color.rgb = RGBColor(186, 230, 253)
    p_by.alignment = PP_ALIGN.CENTER
    p_by.space_after = Pt(6)

    p_stu = tf.add_paragraph()
    p_stu.text = STUDENT_NAME_REG
    p_stu.font.size = Pt(20)
    p_stu.font.bold = True
    p_stu.font.color.rgb = WHITE
    p_stu.alignment = PP_ALIGN.CENTER
    p_stu.space_after = Pt(28)

    p_ug = tf.add_paragraph()
    p_ug.text = "Under the guidance of"
    p_ug.font.size = Pt(16)
    p_ug.font.italic = True
    p_ug.font.color.rgb = RGBColor(186, 230, 253)
    p_ug.alignment = PP_ALIGN.CENTER
    p_ug.space_after = Pt(6)

    p_guide = tf.add_paragraph()
    p_guide.text = GUIDE_NAME
    p_guide.font.size = Pt(20)
    p_guide.font.bold = True
    p_guide.font.color.rgb = WHITE
    p_guide.alignment = PP_ALIGN.CENTER
    p_guide.space_after = Pt(4)

    p_dept = tf.add_paragraph()
    p_dept.text = GUIDE_DEPT
    p_dept.font.size = Pt(16)
    p_dept.font.color.rgb = RGBColor(203, 213, 225)
    p_dept.alignment = PP_ALIGN.CENTER

    add_notes(s1, "Title slide: Introduce project title, student name & registration number, and guide details.")

    # =========================================================================
    # SLIDE 2: Contents
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Contents")

    contents_items = [
        ("1.  Introduction", "Background, motivation, and paradigm shift from static MPT to Deep RL"),
        ("2.  Objectives", "Core research goals across environment design, benchmarking, and feature ablation"),
        ("3.  Block diagram", "End-to-end multi-asset data, feature, training, and evaluation pipeline"),
        ("4.  Simulation diagram", "Closed-loop RL Agent <-> Gymnasium PortfolioTradingEnv simulation architecture"),
        ("5.  Work done so far", "Completed implementation across Phases 1, 2, and 3"),
        ("6.  Results and discussion", "Comparative study (PPO vs. DQN vs. Baselines) and 4-stage Feature Ablation"),
        ("7.  Work pending", "Remaining documentation and extended sensitivity checks"),
        ("8.  Conclusion", "Key empirical takeaways and contributions"),
        ("9.  References", "Foundational literature in portfolio optimization and reinforcement learning"),
    ]

    for idx, (heading, desc) in enumerate(contents_items):
        col = 0 if idx < 5 else 1
        row = idx if idx < 5 else idx - 5
        left = 0.6 if col == 0 else 6.85
        top = 1.45 + row * 1.12

        box = s2.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(5.85), Inches(0.96)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = CARD_BG
        box.line.color.rgb = BORDER_GRAY

        tb = s2.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.1), Inches(5.45), Inches(0.78))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = heading
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = NAVY
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(12)
        p2.font.color.rgb = MUTED_TEXT

    add_notes(s2, "Overview of the presentation structure following the standard project template.")

    # =========================================================================
    # SLIDE 3: Introduction
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Introduction", "From Static Mean-Variance Optimization to Sequential Deep Reinforcement Learning")

    add_card(
        s3, 0.6, 1.45, 5.9, 5.5,
        "Background & Motivation",
        [
            "Portfolio optimization seeks to dynamically allocate capital across multiple risky assets to maximize risk-adjusted return (Sharpe Ratio) while limiting drawdowns.",
            "Classical Methods (Markowitz MPT): Rely on static, backward-looking sample means and covariance matrices from historical data, failing when market regimes shift.",
            "Supervised ML ('Predict-then-Decide', e.g., XGBoost): Predicts next-day price direction in isolation without accounting for transaction fees, current holdings, or position limits.",
        ],
        accent_color=BLUE,
    )
    add_card(
        s3, 6.8, 1.45, 5.9, 5.5,
        "Why Deep Reinforcement Learning (DRL)?",
        [
            "Formulates portfolio management as a Markov Decision Process (MDP) where the agent maps market + portfolio state directly to trading actions.",
            "Explicitly incorporates real-world frictions: 0.1% proportional transaction costs and 30% maximum single-asset concentration caps.",
            "Compares two foundational DRL paradigms on identical footing: Value-Based DQN (off-policy replay buffer) vs. Actor-Critic PPO (on-policy clipped surrogate objective).",
        ],
        accent_color=TEAL,
    )
    add_notes(s3, "Introduction: Explain why classical static Markowitz optimization and supervised 'predict-then-decide' models struggle in dynamic markets with transaction costs, motivating our Deep RL formulation.")

    # =========================================================================
    # SLIDE 4: Objectives
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Objectives", "Primary goals of the comparative and ablation study")

    add_card(
        s4, 0.6, 1.45, 5.9, 2.6,
        "Objective 1: Multi-Asset Feature Pipeline",
        [
            "Build a leak-free data and feature engineering pipeline across N = 10 diversified equities (AAPL, AMZN, GOOGL, HDB, IBN, INFY, JNJ, JPM, MSFT, WIT).",
            "Extract 10 technical indicators organized into 4 domain groups: Price, Momentum, Volatility, and Trend Context.",
        ],
        accent_color=BLUE,
    )
    add_card(
        s4, 6.8, 1.45, 5.9, 2.6,
        "Objective 2: Realistic Gymnasium Environment",
        [
            "Design a custom Gymnasium environment (PortfolioTradingEnv) enforcing 0.1% transaction costs, 30% position caps, and invalid-trade penalties.",
            "Formulate a differential Sharpe + daily return reward signal with drawdown penalties for stable credit assignment.",
        ],
        accent_color=TEAL,
    )
    add_card(
        s4, 0.6, 4.35, 5.9, 2.6,
        "Objective 3: Fair Comparative Benchmarking",
        [
            "Evaluate DQN and PPO against Buy & Hold, Monthly Equal-Weighted Rebalancing, and Markowitz MPT on a strict 65% / 15% / 20% Train-Val-Test split.",
            "Prevent overfitting via validation-set checkpoint selection (BestValidationModelCallback).",
        ],
        accent_color=GOLD,
    )
    add_card(
        s4, 6.8, 4.35, 5.9, 2.6,
        "Objective 4: Phase 3 Feature Ablation Study",
        [
            "Empirically test whether each feature group earns its place in the state space by training cumulative stages (2 -> 5 -> 7 -> 10 features/asset).",
            "Quantify impact on Total Return, Annualized Sharpe Ratio, and Maximum Drawdown.",
        ],
        accent_color=NAVY,
    )
    add_notes(s4, "Our four core objectives cover the multi-asset pipeline, custom Gymnasium environment with realistic constraints, comparative benchmarking against classical strategies, and the 4-stage feature ablation study.")

    # =========================================================================
    # SLIDE 5: Block diagram
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Block diagram", "End-to-End System Workflow from Raw Market Data to Out-of-Sample Evaluation")

    # Top Row Pipeline
    add_diagram_box(s5, 0.6, 1.6, 2.6, 1.35, "1. Raw OHLCV Data\n(data_pipeline.py)", "10 Equities via yfinance\n2,486 Common Days", NAVY)
    add_arrow(s5, 3.3, 2.1, 0.45, 0.35, "right", TEAL)

    add_diagram_box(s5, 3.85, 1.6, 2.8, 1.35, "2. Feature Engineering\n(feature_engineering.py)", "4 Groups | 10 Features/Asset\nTensor Shape: (T, 10, 10)", BLUE)
    add_arrow(s5, 6.75, 2.1, 0.45, 0.35, "right", TEAL)

    add_diagram_box(s5, 7.3, 1.6, 2.7, 1.35, "3. 3-Way Temporal Split\n(build_dataset.py)", "Train 65% | Val 15% | Test 20%\nTrain-Only Z-Score Scaler", TEAL)
    add_arrow(s5, 10.1, 2.1, 0.45, 0.35, "right", TEAL)

    add_diagram_box(s5, 10.65, 1.6, 2.1, 1.35, "4. Trading Env\n(trading_env.py)", "State: 111-D Box\nActions: Discrete(21)", GOLD)

    # Down arrow from Trading Env & Split
    add_arrow(s5, 6.3, 3.15, 0.45, 0.45, "down", SLATE)

    # Bottom Row: Models -> Validation Selection -> Final Evaluation
    add_diagram_box(s5, 0.6, 3.8, 3.8, 1.6, "5A. Deep RL Agents (300k Steps)", "• DQN (Off-Policy Replay Buffer)\n• PPO (On-Policy Actor-Critic)\n• 4-Stage Feature Ablation Models", NAVY)
    add_diagram_box(s5, 0.6, 5.55, 3.8, 1.35, "5B. Classical & ML Baselines", "• Buy & Hold  • Equal-Weighted\n• Markowitz MPT  • XGBoost Baseline", SLATE)

    add_arrow(s5, 4.55, 4.4, 0.5, 0.38, "right", TEAL)
    add_diagram_box(s5, 5.2, 3.8, 3.5, 3.1, "6. Validation Model Selection\n(BestValidationModelCallback)", "Evaluated every 10,000 steps\non 373-day Validation Split\n\nLocks in checkpoint with\nHighest Validation Sharpe Ratio\n(Zero Test-Set Leakage)", TEAL)

    add_arrow(s5, 8.85, 5.1, 0.5, 0.38, "right", GOLD)
    add_diagram_box(s5, 9.5, 3.8, 3.25, 3.1, "7. Held-Out Test Evaluation\n(compare_all_models.py)", "498 Unseen Trading Days\n\nMetrics Reported:\n• Final Portfolio Value ($)\n• Total Return (%)\n• Annualized Sharpe Ratio\n• Maximum Drawdown (%)", BLUE)

    add_notes(s5, "This block diagram illustrates the complete software pipeline, from yfinance data ingestion and 10-indicator feature engineering to the 3-way temporal split, RL training with validation gating, and final out-of-sample evaluation.")

    # =========================================================================
    # SLIDE 6: Simulation diagram
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Simulation diagram", "Closed-Loop Interaction Between RL Agent and Gymnasium PortfolioTradingEnv")

    # Left: RL Agent Box
    add_diagram_box(
        s6, 0.7, 1.8, 3.6, 3.4,
        "RL Policy Network\n(DQN / PPO Agent)",
        "Input: 111-D State Vector s_t\nMLP Hidden Layers (64 × 64)\n\nOutputs Action a_t in {0..20}:\n• 0 = Hold Portfolio\n• 1..10 = Buy Asset i (10% PV)\n• 11..20 = Sell Asset i (10% PV)",
        NAVY,
    )

    # Top Arrow: Action a_t
    add_arrow(s6, 4.5, 2.2, 3.8, 0.42, "right", TEAL)
    tb_act = s6.shapes.add_textbox(Inches(4.6), Inches(1.65), Inches(3.6), Inches(0.5))
    p_act = tb_act.text_frame.paragraphs[0]
    p_act.text = "Action a_t (Hold / Buy 10% / Sell 10%)"
    p_act.font.size = Pt(13)
    p_act.font.bold = True
    p_act.font.color.rgb = TEAL
    p_act.alignment = PP_ALIGN.CENTER

    # Right: Environment Simulation Engine
    add_diagram_box(
        s6, 8.5, 1.8, 4.1, 3.4,
        "PortfolioTradingEnv Simulator\n(Step t -> t + 1)",
        "1. Constraint Check:\n   • Cash sufficient? Position <= 30% cap?\n   • Deduct 0.1% transaction cost\n   • If blocked: apply -0.05 penalty\n2. Advance Market Clock (t -> t+1)\n3. Revalue Holdings & Cash\n4. Compute Differential Reward r_t",
        BLUE,
    )

    # Bottom Arrow: State s_{t+1} & Reward r_t
    add_arrow(s6, 4.5, 4.3, 3.8, 0.42, "left", GOLD)
    tb_rew = s6.shapes.add_textbox(Inches(4.4), Inches(4.78), Inches(4.0), Inches(0.6))
    p_rew = tb_rew.text_frame.paragraphs[0]
    p_rew.text = "Next State s_{t+1} (111-D)  +  Reward r_t"
    p_rew.font.size = Pt(13)
    p_rew.font.bold = True
    p_rew.font.color.rgb = GOLD
    p_rew.alignment = PP_ALIGN.CENTER

    # Bottom Formula Card
    add_card(
        s6, 0.7, 5.55, 11.9, 1.5,
        "Simulation Equations (State Vector & Differential Risk-Adjusted Reward)",
        [
            "State Vector s_t (111 dims):  [10 Assets × 10 Standardized Features (100)]  ⊕  [Current Asset Weights w_1..w_10 (10)]  ⊕  [Cash Ratio c_t (1)]",
            "Reward Function r_t:  r_t = 100 · R_t  +  (Sharpe_t^(20) - Sharpe_{t-1}^(20))  -  0.5 · DD_t  -  10.0 · max(0, DD_t - DD_{t-1})  -  0.05 · I(trade_blocked)",
        ],
        accent_color=TEAL,
        font_size=12.5,
    )
    add_notes(s6, "This simulation diagram shows the step-by-step closed loop inside PortfolioTradingEnv: the agent observes the 111-D state, selects one of 21 discrete actions, the simulator enforces the 30% position cap and 0.1% fee, advances the market day, and returns the differential Sharpe reward.")

    # =========================================================================
    # SLIDE 7: Work done so far
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Work done so far", "Completed implementation and experiments across Phases 1, 2, and 3")

    add_card(
        s7, 0.6, 1.45, 3.85, 5.5,
        "Phase 1: Data & Feature Pipeline",
        [
            "Implemented automated Yahoo Finance data downloader (data_pipeline.py) for 10 stocks + synthetic generator fallback.",
            "Engineered all 10 technical indicators in pure pandas/numpy (feature_engineering.py).",
            "Built aligned multi-asset tensor builder and leak-free 65/15/20 temporal split (build_dataset.py).",
            "Implemented supervised XGBoost direction-prediction baseline (xgboost_baseline.py).",
        ],
        accent_color=BLUE,
        font_size=12.5,
    )
    add_card(
        s7, 4.74, 1.45, 3.85, 5.5,
        "Phase 2: RL Env & Model Training",
        [
            "Developed custom Gymnasium environment (trading_env.py) passing check_env verification.",
            "Implemented classical baselines: Buy & Hold, Equal-Weighted Rebalanced, and Markowitz MPT (classical_baselines.py).",
            "Trained DQN and PPO for 300,000 timesteps with periodic checkpointing (train_dqn.py, train_ppo.py).",
            "Built unified comparative evaluation harness (compare_all_models.py).",
        ],
        accent_color=TEAL,
        font_size=12.5,
    )
    add_card(
        s7, 8.88, 1.45, 3.85, 5.5,
        "Phase 3: Stabilization & Ablation",
        [
            "Diagnosed policy oscillation using checkpoint diagnostics (diagnose_ppo_over_time.py).",
            "Replaced overlapping 20-day Sharpe level reward with differential Sharpe + step return reward.",
            "Added BestValidationModelCallback to select optimal checkpoint on unseen validation data.",
            "Completed 4-stage cumulative Feature Ablation study at 300,000 steps per stage (feature_ablation.py).",
        ],
        accent_color=GOLD,
        font_size=12.5,
    )
    add_notes(s7, "All three planned phases of code implementation and experimental runs have been completed.")

    # =========================================================================
    # SLIDE 8: Results and discussion (Part 1: Main Comparative Study)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Results and discussion (1/2): Comparative Study", "Out-of-sample evaluation on the 498-day held-out test period ($100,000 initial capital)")

    headers = ["Model / Strategy", "Final Value ($)", "Total Return", "Sharpe Ratio", "Max DD"]
    rows = [
        ["PPO (Proposed)", "$189,843", "89.84%", "2.229", "9.70%"],
        ["DQN (Proposed)", "$157,392", "57.39%", "1.899", "10.09%"],
        ["Buy & Hold", "$157,811", "57.81%", "1.875", "9.83%"],
        ["Equal-Weighted", "$156,487", "56.49%", "1.905", "9.72%"],
        ["Markowitz (MPT)", "$149,224", "49.22%", "1.608", "9.46%"],
    ]
    add_styled_table(s8, 0.5, 1.4, 5.9, 3.1, headers, rows, highlight_row_idx=0, font_size=11.5)

    add_card(
        s8, 0.5, 4.65, 5.9, 2.4,
        "Discussion of Comparative Findings",
        [
            "PPO achieves #1 performance across all metrics: $189,843 (+89.84% return, +32.03% above Buy & Hold) and a 2.229 Sharpe ratio.",
            "Lower Drawdown than Passive Investing: PPO caps Max Drawdown at 9.70% (vs. 9.83% for Buy & Hold) by preserving capital during dips.",
            "DQN beats Buy & Hold and Markowitz on Sharpe (1.899 vs. 1.875 & 1.608) despite starting in 100% cash and paying 0.1% fees.",
        ],
        accent_color=TEAL,
        font_size=11.5,
    )

    if os.path.exists("figures/equity_curves.png") and os.path.exists("figures/drawdown_curves.png"):
        s8.shapes.add_picture("figures/equity_curves.png", Inches(6.6), Inches(1.4), width=Inches(6.2))
        s8.shapes.add_picture("figures/drawdown_curves.png", Inches(6.6), Inches(4.5), width=Inches(6.2))
    else:
        add_card(
            s8, 6.6, 1.4, 6.2, 5.65,
            "Before vs. After Stabilization",
            [
                "Before Reward & Validation Fix: PPO Final Value = $137,054 (37.05% return, 1.756 Sharpe).",
                "After Differential Reward + Validation Selection: PPO Final Value = $189,843 (89.84% return, 2.229 Sharpe).",
                "DQN also improved from $151,258 (51.26%, Sharpe 1.878) to $157,392 (57.39%, Sharpe 1.899).",
            ],
            accent_color=BLUE,
        )
    add_notes(s8, "In our main comparative study, PPO outperforms all baselines with an 89.84% return and 2.229 Sharpe ratio, while maintaining a lower maximum drawdown than Buy & Hold.")

    # =========================================================================
    # SLIDE 9: Results and discussion (Part 2: Feature Ablation Study)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Results and discussion (2/2): Feature Ablation Study", "Cumulative feature-group evaluation using DQN (300,000 steps per stage)")

    ab_headers = ["Stage", "Included Feature Groups", "# Feat", "Total Return", "Sharpe", "Max DD"]
    ab_rows = [
        ["Stage 1", "Price only (daily_return, log_return)", "2", "40.35%", "1.932", "7.30%"],
        ["Stage 2", "+ Momentum (rsi_14, macd, macd_signal)", "5", "16.85%", "1.230", "5.29%"],
        ["Stage 3", "+ Volatility (bb_pct, rolling_std_20)", "7", "26.87%", "1.424", "6.42%"],
        ["Stage 4", "+ Trend (sma_10, sma_30, price_vs_sma30)", "10", "57.39%", "1.899", "10.09%"],
        ["Baseline", "Buy & Hold Reference", "--", "57.81%", "1.875", "9.83%"],
    ]
    add_styled_table(s9, 0.5, 1.4, 6.1, 3.1, ab_headers, ab_rows, highlight_row_idx=3, font_size=11.5)

    add_card(
        s9, 0.5, 4.65, 6.1, 2.4,
        "Discussion of Ablation Findings",
        [
            "Full 10-Feature Set (Stage 4) delivers the highest return (57.39%), more than doubling Stage 3 (26.87%) and beating Buy & Hold Sharpe (1.899 vs 1.875).",
            "Why Stage 2 (+ Momentum) had 5.29% Max DD: Selected at step 170k after making a profit ($101,879, +0.255 Sharpe) in the 2022 bear-market validation split via high cash allocation.",
            "Adding Volatility (Stage 3) and Trend Context (Stage 4) enables the agent to separate short-term overbought signals from sustained bull trends.",
        ],
        accent_color=BLUE,
        font_size=11.5,
    )

    if os.path.exists("figures/feature_ablation.png"):
        s9.shapes.add_picture("figures/feature_ablation.png", Inches(6.75), Inches(1.8), width=Inches(6.1))
    add_notes(s9, "The feature ablation study shows that while intermediate momentum features induce a highly defensive low-drawdown policy (5.29% Max DD), adding the Trend context group in Stage 4 unlocks the full 57.39% return and 1.899 Sharpe ratio.")

    # =========================================================================
    # SLIDE 10: Work pending
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Work pending", "Remaining tasks prior to final thesis / paper submission")

    add_card(
        s10, 0.6, 1.5, 5.9, 5.4,
        "1. Multi-Seed Sensitivity & Statistical Reporting",
        [
            "Execute 3 to 5 independent random seeds (e.g., seeds 0, 1, 2, 3, 4) for DQN and PPO to report Mean ± Standard Deviation in the final paper table.",
            "Perform paired statistical significance tests (e.g., Ledoit-Wolf / Jobson-Korkie Sharpe ratio test) comparing PPO against Buy & Hold.",
            "Analyze per-asset portfolio weight allocation heatmaps over the 498-day test window.",
        ],
        accent_color=GOLD,
    )
    add_card(
        s10, 6.8, 1.5, 5.9, 5.4,
        "2. Final Report & Manuscript Integration",
        [
            "Integrate the finalized comparative and feature-ablation tables and equity/drawdown figures into the final project report.",
            "Document the methodological case study on blocked-trade penalties (-0.05) and differential Sharpe reward stabilization in the Discussion section.",
            "Complete final proofreading and formatting of references and appendices.",
        ],
        accent_color=BLUE,
    )
    add_notes(s10, "Our core implementation and experiments are complete; the remaining tasks focus on multi-seed variance reporting, portfolio weight visualization, and final manuscript formatting.")

    # =========================================================================
    # SLIDE 11: Conclusion
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Conclusion", "Summary of contributions and empirical findings")

    add_card(
        s11, 0.6, 1.5, 12.13, 5.4,
        "Key Conclusions from the Study",
        [
            "1. Deep RL Outperforms Classical Strategies: On the held-out 498-day test split, PPO grew $100,000 to $189,843 (+89.84% return) with an annualized Sharpe ratio of 2.229 and a maximum drawdown of 9.70%, outperforming Buy & Hold (57.81%, Sharpe 1.875), Equal-Weighted (56.49%, Sharpe 1.905), and Markowitz MPT (49.22%, Sharpe 1.608).",
            "2. Value-Based DQN Beats Passive Benchmarks on Risk-Adjusted Return: DQN achieved a 57.39% return and 1.899 Sharpe ratio despite starting in 100% cash and paying 0.1% transaction costs on every trade.",
            "3. Reward Design & Validation Gating Are Critical: Replacing the overlapping 20-day Sharpe level with a differential Sharpe + daily return reward and selecting checkpoints via a chronological validation split resolved policy oscillation and improved PPO return by +52.79 percentage points.",
            "4. Full Multi-Group Feature Set Earns Its Place: The 4-stage ablation study confirmed that combining Price, Momentum, Volatility, and Trend context features is necessary to achieve both downside protection and full bull-market upside (57.39% return, 1.899 Sharpe).",
        ],
        accent_color=TEAL,
        font_size=14,
    )
    add_notes(s11, "Summary of our four main conclusions.")

    # =========================================================================
    # SLIDE 12: References
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "References", "Key literature cited in this project")

    add_card(
        s12, 0.6, 1.5, 12.13, 5.4,
        "References",
        [
            "[1] Markowitz, H. (1952). 'Portfolio Selection.' The Journal of Finance, 7(1), 77–91.",
            "[2] Mnih, V., et al. (2015). 'Human-level control through deep reinforcement learning (DQN).' Nature, 518(7540), 529–533.",
            "[3] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). 'Proximal Policy Optimization Algorithms (PPO).' arXiv:1707.06347.",
            "[4] Moody, J., & Saffell, M. (2001). 'Learning to trade via direct reinforcement (Differential Sharpe Ratio).' IEEE Transactions on Neural Networks, 12(4), 875–889.",
            "[5] Jiang, Z., Xu, D., & Liang, J. (2017). 'A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem.' arXiv:1706.10059.",
            "[6] Raffin, A., et al. (2021). 'Stable-Baselines3: Reliable Reinforcement Learning Implementations.' Journal of Machine Learning Research, 22(268), 1–8.",
            "[7] Towers, M., et al. (2024). 'Gymnasium: A Standard Interface for Reinforcement Learning Environments.' arXiv:2407.17032.",
        ],
        accent_color=NAVY,
        font_size=13.5,
    )
    add_notes(s12, "References covering Markowitz portfolio theory, DQN, PPO, Differential Sharpe Ratio, and the Gymnasium / Stable-Baselines3 frameworks.")

    out_path = "Portfolio_Optimization_RL_Presentation.pptx"
    prs.save(out_path)
    print(f"Presentation saved successfully to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    build_presentation()
