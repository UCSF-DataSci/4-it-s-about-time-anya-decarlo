def preprocess_data(data, output_dir='data/processed'):
    """Clean and prepare the physiological data for analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw physiological data
    output_dir : str
        Directory to save processed data files
        
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- normalise column names ------------------------------------------------
    df = data.copy()
    df.columns = [c.lower() for c in df.columns]

    # locate key columns (case-insensitive)
    t_col = next(c for c in df.columns if c.lower() == "timestamp")
    sid_col = next(c for c in df.columns if c.lower() == "subject_id")
    sess_col = next(c for c in df.columns if c.lower() == "session")
    numeric_cols = [c for c in ("heart_rate", "eda", "temperature") if c in df.columns]

    # ensure timestamp dtype is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[t_col]):
        df[t_col] = pd.to_datetime(df[t_col], unit="s", errors="coerce")

    all_processed = []

    # process each (subject, session) independently
    for (sid, sess), grp in df.groupby([sid_col, sess_col]):
        g = grp.set_index(t_col).sort_index()

        # 1. resample to 1-second grid (mean aggregation)
        g = g.resample("1S").mean()

        # 2. initial imputation
        g[numeric_cols] = g[numeric_cols].interpolate(limit_direction="both")
        g[numeric_cols] = g[numeric_cols].ffill().bfill()

        # 3. outlier removal + re-impute
        for col in numeric_cols:
            z = zscore(g[col].dropna())
            out_idx = g[col].dropna().index[np.abs(z) > 3.5]
            g.loc[out_idx, col] = np.nan
            g[col] = g[col].interpolate(limit_direction="both").ffill().bfill()

        # restore id/session columns for concat
        g[sid_col] = sid
        g[sess_col] = sess
        all_processed.append(g)

        # 4. save per-subject file
        out_path = Path(output_dir) / f"{sid}_processed.csv"
        g.reset_index().to_csv(out_path, index=False)

    # concat all groups and return tidy frame
    processed = (
        pd.concat(all_processed).reset_index().rename(columns={"index": "timestamp"})
        if all_processed
        else pd.DataFrame(
            columns=["timestamp", "heart_rate", "eda", "temperature", "subject_id", "session"]
        )
    )

    return processed