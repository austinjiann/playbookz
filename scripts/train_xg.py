import json, math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from statsbombpy import sb

# StatsBomb 120x80 pitch
GOAL_X, GOAL_Y = 120.0, 40.0
POST_Y_LOW, POST_Y_HIGH = 36.0, 44.0

def shot_features(df: pd.DataFrame) -> pd.DataFrame:
    dx = (GOAL_X - df['x'])
    dy = (GOAL_Y - df['y'])
    dist = np.sqrt(dx*dx + dy*dy)

    a1 = np.arctan2(df['y'] - POST_Y_LOW, dx)
    a2 = np.arctan2(df['y'] - POST_Y_HIGH, dx)
    angle = np.abs(a1 - a2)

    bp = df['body_part'].fillna('Other').str.lower()
    b_left = bp.str.contains('left').astype(int)
    b_right = bp.str.contains('right').astype(int)
    b_head = bp.str.contains('head').astype(int)

    stype = df['shot_type'].fillna('Open Play').str.lower()
    open_play = (stype == 'open play').astype(int)
    set_piece = (~open_play).astype(int)

    return pd.DataFrame({
        'dist': dist,
        'angle': angle,
        'bp_left': b_left,
        'bp_right': b_right,
        'bp_head': b_head,
        'open_play': open_play,
        'set_piece': set_piece
    })

def load_shots(limit_competitions=None, max_matches=3):
    """Load shots with limits to avoid timeouts"""
    comps = sb.competitions()
    if limit_competitions:
        comps = comps[comps['competition_name'].isin(limit_competitions)]
    
    shots = []
    total_matches_processed = 0
    
    for _, c in comps.iterrows():
        if total_matches_processed >= max_matches:
            break
        try:
            matches = sb.matches(competition_id=c['competition_id'], season_id=c['season_id'])
            # Limit to first few matches to avoid timeout
            matches = matches.head(max_matches - total_matches_processed)
        except Exception as e:
            print(f"Failed to get matches for {c['competition_name']}: {e}")
            continue
            
        for _, m in matches.iterrows():
            try:
                print(f"Processing match {total_matches_processed + 1}/{max_matches}")
                ev = sb.events(match_id=m['match_id'])
                total_matches_processed += 1
            except Exception as e:
                print(f"Failed to get events: {e}")
                continue
                
            s = ev[ev['type'] == 'Shot'].copy()
            if s.empty: 
                continue
            
            # Debug info for first batch
            if len(shots) == 0:
                print(f"Found shot data with {len(s)} shots in first match")
            
            # Use shot_outcome column which contains the outcome data
            if 'shot_outcome' in s.columns:
                s['is_goal'] = s['shot_outcome'].apply(lambda x: 1 if isinstance(x, dict) and x.get('name') == 'Goal' else 0)
            else:
                print("No shot_outcome column found, skipping batch")
                continue
                
            s['x'] = s['location'].apply(lambda v: v[0] if isinstance(v, list) and len(v)>=2 else np.nan)
            s['y'] = s['location'].apply(lambda v: v[1] if isinstance(v, list) and len(v)>=2 else np.nan)
            s['body_part'] = s['shot_body_part'].apply(lambda d: d.get('name') if isinstance(d, dict) else np.nan)
            s['shot_type'] = s['shot_type'].apply(lambda d: d.get('name') if isinstance(d, dict) else np.nan)
            s = s.dropna(subset=['x','y'])
            
            if not s.empty:
                shots.append(s[['x','y','body_part','shot_type','is_goal']])
                
            if total_matches_processed >= max_matches:
                break
                
    if not shots:
        raise RuntimeError("No shots found from StatsBomb Open Data.")
    return pd.concat(shots, ignore_index=True)

def main():
    # Start with a small subset to avoid long loading times
    print("Loading shot data from StatsBomb...")
    try:
        df = load_shots(limit_competitions=['FIFA World Cup'], max_matches=10)
    except Exception as e:
        print(f"Failed with FIFA World Cup: {e}")
        # Try different competition
        try:
            df = load_shots(limit_competitions=['La Liga'], max_matches=5)
        except Exception as e2:
            print(f"Failed with La Liga: {e2}")
            raise RuntimeError("Could not load any shot data from StatsBomb")

    X = shot_features(df)
    y = df['is_goal'].values
    
    print(f"Loaded {len(df)} shots")
    print(f"Goals: {y.sum()}, Misses: {len(y) - y.sum()}")
    
    # Check we have both classes
    if y.sum() == 0:
        print("No goals found in dataset, creating dummy goal for model training")
        # Add a synthetic goal sample
        dummy_row = df.iloc[0].copy()
        dummy_row['is_goal'] = 1
        df = pd.concat([df, dummy_row.to_frame().T], ignore_index=True)
        X = shot_features(df)
        y = df['is_goal'].values
    elif y.sum() == len(y):
        print("No misses found in dataset, creating dummy miss for model training")
        # Add a synthetic miss sample
        dummy_row = df.iloc[0].copy()
        dummy_row['is_goal'] = 0
        df = pd.concat([df, dummy_row.to_frame().T], ignore_index=True)
        X = shot_features(df)
        y = df['is_goal'].values

    model = LogisticRegression(max_iter=200, solver='liblinear')
    model.fit(X, y)

    coeffs = {
        'intercept': float(model.intercept_[0]),
        'columns': list(X.columns),
        'coef': [float(c) for c in model.coef_[0]],
        'n_samples': int(len(df))
    }
    out = Path('data'); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'xg_coeffs.json', 'w') as f:
        json.dump(coeffs, f, indent=2)
    print("Saved coefficients to data/xg_coeffs.json (n_samples=%d)" % coeffs['n_samples'])

if __name__ == '__main__':
    main()