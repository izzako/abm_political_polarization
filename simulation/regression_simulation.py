from dotenv import load_dotenv
load_dotenv()

import logging
import pandas as pd
import numpy as np
import os
import sys
import re
import json
import argparse
from datetime import datetime, timedelta

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import src.const as srconst


# =============================================================================
# HELPER: Per-agent ground-truth opinion time series
# =============================================================================

def get_real_historical(user, sim_interaction_data, sim_text_data,
                        start=pd.to_datetime("2024-02-01 00:00:00")):
    """
    Build a daily opinion time series for a single agent.

    Combines original tweets/replies + retweet-inherited sentiment,
    averages before `start` into a single seed value, then produces
    daily means with forward/back-fill.

    Returns:
        df_user_full : DataFrame with columns
            ['datetime', 'sentiment_label', 'incremental_mean']
            covering start → 2024-02-28 (one row per day).
    """
    # --- Retweet-inherited sentiment -----------------------------------------
    retweet_df = sim_interaction_data.loc[
        (sim_interaction_data.source_author == user) &
        (sim_interaction_data.interaction_type == 'retweet'),
        ['datetime', 'target_tweet_id']
    ].merge(
        sim_text_data.reset_index()[['tweet_id', 'sentiment_label']],
        how='left',
        left_on='target_tweet_id',
        right_on='tweet_id'
    )[['datetime', 'tweet_id', 'sentiment_label']].set_index('tweet_id')

    # --- Original tweets / replies -------------------------------------------
    reply_original_df = sim_text_data.loc[
        sim_text_data['author'] == user,
        ['datetime', 'sentiment_label']
    ]

    # Concat non-empty frames
    frames = [df for df in [retweet_df, reply_original_df] if not df.empty]
    if frames:
        df_user = pd.concat(frames).sort_values(by='datetime')
    else:
        df_user = pd.DataFrame(columns=['datetime', 'sentiment_label'])

    # --- Map sentiment to numeric via srconst --------------------------------
    df_user['sentiment_label'] = (
        df_user['sentiment_label'].map(srconst.sentiment_map).astype(float)
    )
    df_user['datetime'] = pd.to_datetime(df_user['datetime'].dt.date)

    end = pd.to_datetime("2024-02-28 00:00:00")
    base_date = pd.to_datetime("2024-02-01 00:00:00")

    # --- Handle init-days collapsing -----------------------------------------
    if start > base_date:
        df_user_before_start = df_user[
            (df_user['datetime'] >= base_date) &
            (df_user['datetime'] < start + pd.Timedelta(days=1))
        ]
        avg_sentiment_before_start = (
            df_user_before_start['sentiment_label'].mean()
            if not df_user_before_start.empty else 0
        )

        df_user_from_start = df_user[df_user['datetime'] >= start + pd.Timedelta(days=1)]
        df_user2 = (
            df_user_from_start.groupby(['datetime'], as_index=False)['sentiment_label'].mean()
            if not df_user_from_start.empty
            else pd.DataFrame(columns=['datetime', 'sentiment_label'])
        )

        df_user_full = pd.DataFrame(
            pd.date_range(start, end), columns=['datetime']
        ).merge(df_user2, how='left', on='datetime')

        df_user_full.loc[0, 'sentiment_label'] = avg_sentiment_before_start
        df_user_full['sentiment_label'] = df_user_full['sentiment_label'].ffill().bfill()

    else:
        df_user2 = (
            df_user.groupby(['datetime'], as_index=False)['sentiment_label'].mean()
            if not df_user.empty
            else pd.DataFrame(columns=['datetime', 'sentiment_label'])
        )
        df_user_full = pd.DataFrame(
            pd.date_range(start, end), columns=['datetime']
        ).merge(df_user2, how='left', on='datetime')
        df_user_full['sentiment_label'] = df_user_full['sentiment_label'].ffill().bfill()

    # Fill any remaining NaN (agent with zero activity) with 0
    df_user_full['sentiment_label'] = df_user_full['sentiment_label'].fillna(0.0)

    df_user_full['incremental_mean'] = (
        df_user_full['sentiment_label'].expanding().mean()
    )

    return df_user_full


def compute_ground_truth_opinions(sim_text_data, sim_interaction_data,
                                  list_agents, time_index, init_days=1):
    """
    Build the opinion matrix by calling get_real_historical for every agent.

    Args:
        sim_text_data      : filtered text DataFrame
        sim_interaction_data : filtered interaction DataFrame
        list_agents        : sorted list of agent names
        time_index         : pd.DatetimeIndex of daily buckets
        init_days          : number of days to collapse into the seed opinion

    Returns:
        opinion_filled : np.ndarray (n_steps, n_agents)  — daily mean sentiment,
                         ffilled/bfilled, ready for regression.
        opinion_raw    : np.ndarray (n_steps, n_agents)  — same but with NaN
                         where agent had no raw activity on that day
                         (used for evaluation only).
        incremental_mean : np.ndarray (n_steps, n_agents) — expanding mean.
    """
    start = pd.to_datetime("2024-02-01 00:00:00") + pd.Timedelta(days=init_days - 1)

    n_agents = len(list_agents)
    n_steps = len(time_index)
    opinion_filled = np.zeros((n_steps, n_agents))
    incremental_mean = np.zeros((n_steps, n_agents))

    for j, user in enumerate(list_agents):
        df_hist = get_real_historical(user, sim_interaction_data, sim_text_data, start=start)

        # Align: get_real_historical returns dates from `start`→Feb 28.
        # time_index may start from Feb 1.  We align by matching dates.
        hist_dates = df_hist['datetime'].dt.normalize()
        ti_dates = time_index.normalize()

        for t, d in enumerate(ti_dates):
            match = hist_dates == d
            if match.any():
                opinion_filled[t, j] = df_hist.loc[match.values, 'sentiment_label'].iloc[0]
                incremental_mean[t, j] = df_hist.loc[match.values, 'incremental_mean'].iloc[0]
            else:
                # Date falls before the agent's historical start → use first available
                opinion_filled[t, j] = df_hist['sentiment_label'].iloc[0]
                incremental_mean[t, j] = df_hist['incremental_mean'].iloc[0]

    # Build a "raw" version where days with no original activity are NaN
    # (useful for evaluation — we only evaluate on days with real data)
    opinion_raw = opinion_filled.copy()
    # Mark days that were purely forward-filled as NaN in raw version
    # We detect this by checking if the agent had any tweets/interactions that day
    sim_text_data_c = sim_text_data.copy()
    sim_interaction_data_c = sim_interaction_data.copy()
    sim_text_data_c['datetime'] = pd.to_datetime(sim_text_data_c['datetime'])
    sim_interaction_data_c['datetime'] = pd.to_datetime(sim_interaction_data_c['datetime'])

    agent_to_idx = {a: i for i, a in enumerate(list_agents)}
    for t, d in enumerate(time_index):
        d_start = d.normalize()
        d_end = d_start + pd.Timedelta(days=1)

        active_text = set(
            sim_text_data_c[
                (sim_text_data_c['datetime'] >= d_start) &
                (sim_text_data_c['datetime'] < d_end)
            ]['author'].unique()
        )
        active_int = set(
            sim_interaction_data_c[
                (sim_interaction_data_c['datetime'] >= d_start) &
                (sim_interaction_data_c['datetime'] < d_end)
            ]['source_author'].unique()
        )
        active_agents = active_text | active_int

        for j, agent in enumerate(list_agents):
            if agent not in active_agents:
                opinion_raw[t, j] = np.nan

    return opinion_filled, opinion_raw, incremental_mean


# =============================================================================
# HELPER: Build binary interaction matrix per time bucket
# =============================================================================

def build_interaction_features(interaction_df, list_agents, time_index):
    """
    For every time bucket, build a binary (n_agents x n_agents) matrix:
        interaction_tensor[t, i, j] = 1 if agent i interacted with agent j
                                       in time bucket t.

    Returns:
        interaction_tensor : np.ndarray of shape (n_steps, n_agents, n_agents)
    """
    interaction_df = interaction_df.copy()
    interaction_df["datetime"] = pd.to_datetime(interaction_df["datetime"])
    agent_to_idx = {a: i for i, a in enumerate(list_agents)}
    n_agents = len(list_agents)
    n_steps = len(time_index)
    tensor = np.zeros((n_steps, n_agents, n_agents), dtype=np.float32)

    for t in range(n_steps):
        t_start = time_index[t]
        t_end = time_index[t + 1] if t + 1 < n_steps else t_start + pd.Timedelta(days=1)

        bucket = interaction_df[
            (interaction_df["datetime"] >= t_start) & (interaction_df["datetime"] < t_end)
        ]
        for _, row in bucket.iterrows():
            src = row["source_author"]
            tgt = row["target_author"]
            if src in agent_to_idx and tgt in agent_to_idx and src != tgt:
                tensor[t, agent_to_idx[src], agent_to_idx[tgt]] = 1.0

    return tensor


# =============================================================================
# HELPER: Assemble per-agent regression dataset
# =============================================================================

def build_agent_dataset(agent_idx, opinion_filled, interaction_tensor):
    """
    For a single agent, build the feature matrix X and target vector y
    over all available time steps (t >= 1, since we need t-1 as features).

    Features per sample (at time t):
        - prev_opinion        : agent's own opinion at t-1
        - prev_opinion_sq     : squared previous opinion (nonlinearity)
        - interaction_j (x N) : binary, did agent interact with agent j at t-1
        - neighbour_opinion_j : opinion of agent j at t-1  (only where interaction = 1,
                                  else 0  — keeps dimensionality constant)
        - mean_neighbour_opinion : mean opinion of interacted neighbours at t-1
        - n_interactions      : how many agents this agent interacted with at t-1

    Target:
        - opinion at time t  (clipped to [-1, 1])
    """
    n_steps, n_agents = opinion_filled.shape
    rows_X = []
    rows_y = []

    for t in range(1, n_steps):
        prev_op = opinion_filled[t - 1, agent_idx]
        interaction_vec = interaction_tensor[t - 1, agent_idx, :]  # shape (n_agents,)
        neighbour_ops = opinion_filled[t - 1, :] * interaction_vec  # 0 where no interaction
        n_int = interaction_vec.sum()
        mean_neigh = neighbour_ops.sum() / n_int if n_int > 0 else 0.0

        features = np.concatenate([
            [prev_op, prev_op ** 2, mean_neigh, n_int],
            interaction_vec,         # binary interaction indicators
            neighbour_ops,           # neighbour opinions (masked)
        ])
        rows_X.append(features)
        rows_y.append(opinion_filled[t, agent_idx])

    return np.array(rows_X), np.array(rows_y)


def build_feature_names(list_agents):
    """Return human-readable feature names matching build_agent_dataset output."""
    n = len(list_agents)
    names = ["prev_opinion", "prev_opinion_sq", "mean_neighbour_opinion", "n_interactions"]
    names += [f"interact_{a}" for a in list_agents]
    names += [f"neigh_op_{a}" for a in list_agents]
    return names


# =============================================================================
# CORE: Train per-agent regression models on the first half of the time range
# =============================================================================

def train_agent_models(opinion_filled, interaction_tensor, train_steps, list_agents, alpha=1.0):
    """
    Train one Ridge regression model per agent on the first `train_steps`
    time steps.

    Returns:
        models   : dict[int -> trained Ridge model]
        scalers  : dict[int -> fitted StandardScaler]
        metrics  : dict[int -> dict with rmse/mae/r2]
    """
    n_agents = opinion_filled.shape[1]
    models = {}
    scalers = {}
    metrics = {}

    for idx in range(n_agents):
        X, y = build_agent_dataset(idx, opinion_filled, interaction_tensor)
        # training portion: first (train_steps - 1) samples since dataset starts at t=1
        n_train = train_steps - 1
        X_train, y_train = X[:n_train], y[:n_train]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)

        model = Ridge(alpha=alpha)
        model.fit(X_train_sc, y_train)

        y_pred = np.clip(model.predict(X_train_sc), -1, 1)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred) if len(set(y_train)) > 1 else 0.0

        models[idx] = model
        scalers[idx] = scaler
        metrics[idx] = {"rmse": rmse, "mae": mae, "r2": r2}

    return models, scalers, metrics


# =============================================================================
# CORE: Simulate (auto-regressive roll-forward) for the test period
# =============================================================================

def simulate_opinions(models, scalers, opinion_filled, interaction_tensor,
                      train_steps, list_agents):
    """
    Starting from the last training time step, roll forward one step at a
    time: predict each agent's opinion, clip to [-1, 1], then feed that
    prediction back as the input for the next step.

    Returns:
        sim_opinions : np.ndarray (n_total_steps, n_agents)
                       The first `train_steps` rows are ground truth;
                       subsequent rows are model predictions.
    """
    n_steps, n_agents = opinion_filled.shape
    sim = opinion_filled.copy()

    for t in range(train_steps, n_steps):
        for idx in range(n_agents):
            prev_op = sim[t - 1, idx]
            interaction_vec = interaction_tensor[t - 1, idx, :]
            neighbour_ops = sim[t - 1, :] * interaction_vec
            n_int = interaction_vec.sum()
            mean_neigh = neighbour_ops.sum() / n_int if n_int > 0 else 0.0

            features = np.concatenate([
                [prev_op, prev_op ** 2, mean_neigh, n_int],
                interaction_vec,
                neighbour_ops,
            ]).reshape(1, -1)

            features_sc = scalers[idx].transform(features)
            pred = models[idx].predict(features_sc)[0]
            sim[t, idx] = np.clip(pred, -1, 1)

    return sim


# =============================================================================
# CORE: Evaluate on the held-out test portion
# =============================================================================

def evaluate_simulation(sim_opinions, ground_truth, train_steps, list_agents):
    """
    Compare simulated opinions against ground truth for the test period.
    Returns per-agent and aggregate metrics.
    """
    n_agents = ground_truth.shape[1]
    gt_test = ground_truth[train_steps:]
    sim_test = sim_opinions[train_steps:]

    agent_metrics = {}
    all_gt, all_sim = [], []

    for idx in range(n_agents):
        mask = ~np.isnan(gt_test[:, idx])
        if mask.sum() == 0:
            continue
        g = gt_test[mask, idx]
        s = sim_test[mask, idx]
        all_gt.extend(g)
        all_sim.extend(s)
        agent_metrics[list_agents[idx]] = {
            "rmse": np.sqrt(mean_squared_error(g, s)),
            "mae": mean_absolute_error(g, s),
        }

    all_gt = np.array(all_gt)
    all_sim = np.array(all_sim)
    aggregate = {
        "rmse": np.sqrt(mean_squared_error(all_gt, all_sim)),
        "mae": mean_absolute_error(all_gt, all_sim),
        "r2": r2_score(all_gt, all_sim) if len(set(all_gt)) > 1 else 0.0,
    }
    return aggregate, agent_metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ---- CLI ----------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    srconst.load_config(args.config)

    # ---- Load data ----------------------------------------------------------
    text_df = pd.read_parquet(srconst.get("text_data", "DATA_PATH"))
    interaction_df = pd.read_parquet(srconst.get("interaction_data", "DATA_PATH"))

    topic_num = srconst.get("topic_num", "SIMULATION")
    if topic_num != "all":
        topic_num = int(topic_num)

    init_days = int(srconst.get("init_days", "SIMULATION"))
    train_cutoff_day = int(srconst.get("train_cutoff_day", "SIMULATION"))
    ridge_alpha = float(srconst.get("ridge_alpha", "MODEL"))
    mode = srconst.get("model", "MODEL")  # 'regression' from config

    EXPERIMENT_OUTPUT_DIR = os.path.join(srconst.OUTPUT_DIR, f"heuristic_{mode}", f"{topic_num}")
    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)

    # ---- Filter by topic ----------------------------------------------------
    if topic_num == "all":
        sim_text_data = text_df
        sim_interaction_data = interaction_df
    else:
        topic_title = srconst.topics[topic_num]
        print(f"Topic {topic_num}: {topic_title}")
        sim_text_data = text_df[text_df["topic_label"] == topic_title].copy()
        sim_interaction_data = interaction_df[
            interaction_df["target_tweet_id"].isin(sim_text_data["tweet_id"])
        ].copy()
        sim_interaction_data = sim_interaction_data[
            ~(
                (sim_interaction_data["interaction_type"] == "reply")
                & ~(sim_interaction_data["source_tweet_id"].isin(sim_text_data.index))
            )
        ]

    # ---- Agent list ---------------------------------------------------------
    list_agents = sorted(
        set(
            sim_text_data["author"].tolist()
            + sim_interaction_data["source_author"].tolist()
            + sim_interaction_data["target_author"].tolist()
        )
    )
    user_to_idx = {user: i for i, user in enumerate(list_agents)}
    print(f"Total agents: {len(list_agents)}")

    # ---- Time index (daily buckets, Feb 1 – Feb 28) -------------------------
    time_index = pd.date_range("2024-02-01", "2024-02-28", freq="1D")
    train_cutoff = pd.Timestamp(f"2024-02-{str(train_cutoff_day).zfill(2)}")
    train_steps = int((train_cutoff - time_index[0]).days)
    # Since build_agent_dataset starts at t=1, effective training samples = train_steps - 1

    print(f"Time buckets: {len(time_index)}")
    print(f"Training buckets (Feb 1-14): {train_steps}")
    print(f"Test buckets (Feb 15-28): {len(time_index) - train_steps}")

    # ---- Ground truth opinions per bucket -----------------------------------
    print("\nComputing ground-truth opinions per time bucket...")
    opinion_filled, opinion_raw, incremental_mean = compute_ground_truth_opinions(
        sim_text_data, sim_interaction_data, list_agents, time_index, init_days=init_days
    )
    print(f"Opinion matrix shape: {opinion_filled.shape}  (steps x agents)")

    # ---- Interaction features per bucket ------------------------------------
    print("Building interaction feature tensors...")
    interaction_tensor = build_interaction_features(
        sim_interaction_data, list_agents, time_index
    )
    print(f"Interaction tensor shape: {interaction_tensor.shape}")

    # ---- Train per-agent Ridge models on first 50% --------------------------
    print("\nTraining per-agent Ridge regression models...")
    models, scalers, train_metrics = train_agent_models(
        opinion_filled, interaction_tensor, train_steps, list_agents, alpha=ridge_alpha
    )

    avg_train_rmse = np.mean([m["rmse"] for m in train_metrics.values()])
    avg_train_r2 = np.mean([m["r2"] for m in train_metrics.values()])
    print(f"  Average training RMSE : {avg_train_rmse:.4f}")
    print(f"  Average training R²   : {avg_train_r2:.4f}")

    # ---- Simulate (auto-regressive) for test period -------------------------
    print("\nSimulating opinions for test period (auto-regressive roll-forward)...")
    sim_opinions = simulate_opinions(
        models, scalers, opinion_filled, interaction_tensor, train_steps, list_agents
    )

    # ---- Evaluate on held-out test period -----------------------------------
    print("\nEvaluating on test period (Feb 15-28)...")
    agg_metrics, agent_metrics = evaluate_simulation(
        sim_opinions, opinion_raw, train_steps, list_agents  # compare against raw (not filled)
    )
    print(f"  Aggregate test RMSE : {agg_metrics['rmse']:.4f}")
    print(f"  Aggregate test MAE  : {agg_metrics['mae']:.4f}")
    print(f"  Aggregate test R²   : {agg_metrics['r2']:.4f}")

    # ---- Build output DataFrame (matching original code format) ---------------
    # Original format:
    #   date_0 = '2024-02-{init_days} 21:36:00'  (first row = averaged seed opinion)
    #   x_date = 2.4h freq from (date_0 + 1 day) to end_date
    #   columns after melt: datetime, topic, agent, opinion_weight

    start = datetime.strptime('2024-02-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_date = '2024-02-28'
    date_0 = pd.Timestamp(f'2024-02-{str(init_days).zfill(2)} 21:36:00')
    x_date = pd.date_range((date_0 + pd.Timedelta(days=1)).date(), end_date, freq='2.4h')
    step = len(x_date)

    # First row = averaged seed opinion (opinion at init_days, i.e. the collapsed
    # average from get_real_historical — this is sim_opinions[train_steps - 1])
    X0 = sim_opinions[train_steps - 1]  # Feb 14 = the init-days seed row

    # Build the simulation history list matching original structure:
    #   hist_X[0] = X0  (seed)
    #   hist_X[1..step] = predictions at each 2.4h step
    # Since regression is daily, we interpolate daily predictions into 2.4h slots
    # by holding each day's prediction constant within that day.
    daily_predictions = sim_opinions[train_steps:]  # Feb 15 → Feb 28, shape (14, n_agents)
    daily_dates = time_index[train_steps:]           # Feb 15 → Feb 28

    hist_X = [X0]
    for x_dt in x_date:
        # Find which daily bucket this 2.4h timestamp falls into
        day_date = x_dt.normalize()
        day_idx = np.searchsorted(daily_dates, day_date, side='right') - 1
        day_idx = np.clip(day_idx, 0, len(daily_predictions) - 1)
        hist_X.append(daily_predictions[day_idx])

    x_date_viz = ([date_0] + list(x_date))[:len(hist_X)]
    out_df = pd.DataFrame(np.array(hist_X)).rename(
        {j: i for i, j in user_to_idx.items()}, axis=1
    )
    out_df['datetime'] = x_date_viz
    out_df['topic'] = topic_num
    out_df2 = (
        out_df.melt(id_vars=['datetime', 'topic'], var_name='agent', value_name='opinion_weight')
        .sort_values(by=['datetime', 'agent'], ignore_index=True)
    )
    out_df2.rename(columns={'datetime':'time_step'}, inplace=True)

    # ---- Save outputs (matching original filenames & formats) ----------------
    out_df2.to_csv(
        os.path.join(EXPERIMENT_OUTPUT_DIR, f'opinion_shift_step_{step}.csv'),
        index=False, sep=';', quoting=1,
    )
    json.dump(
        user_to_idx,
        open(os.path.join(EXPERIMENT_OUTPUT_DIR, 'user_to_idx.json'), 'w'),
        indent=4,
    )

    # Save interaction graph (matching original: source_author, target_author, weight)
    sim_interaction_data_c = sim_interaction_data.copy()
    sim_interaction_data_c['datetime'] = pd.to_datetime(sim_interaction_data_c['datetime'])
    simulation_start = start + pd.Timedelta(days=init_days)
    init_interactions = sim_interaction_data_c[sim_interaction_data_c['datetime'] < simulation_start]

    # Build edge weights from init-period interactions (binary count as weight)
    edges = init_interactions.groupby(
        ['source_author', 'target_author'], as_index=False
    ).size().rename(columns={'size': 'weight'})
    edges = edges.loc[edges['source_author'] != edges['target_author']]
    edges.to_csv(
        os.path.join(EXPERIMENT_OUTPUT_DIR, 'interaction_graph.csv'),
        index=False, sep=';', quoting=1,
    )

    # Save metrics
    metrics_out = {"aggregate_test": agg_metrics, "aggregate_train_rmse": avg_train_rmse}
    json.dump(
        metrics_out,
        open(os.path.join(EXPERIMENT_OUTPUT_DIR, 'metrics.json'), 'w'),
        indent=4,
    )

    # Save per-agent test metrics
    pd.DataFrame(agent_metrics).T.to_csv(
        os.path.join(EXPERIMENT_OUTPUT_DIR, 'agent_test_metrics.csv'), sep=';'
    )

    print(f"\nOutputs saved to: {EXPERIMENT_OUTPUT_DIR}")
    print(f"  opinion_shift_step_{step}.csv")
    print(f"  user_to_idx.json")
    print(f"  interaction_graph.csv")
    print(f"  metrics.json")
    print(f"  agent_test_metrics.csv")
    print("Done.")


if __name__ == "__main__":
    main()