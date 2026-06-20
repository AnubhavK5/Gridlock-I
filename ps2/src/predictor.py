import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings("ignore")

CSV_PATH = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"

EVENT_TYPES   = ["unplanned", "planned"]
EVENT_CAUSES  = ["vehicle_breakdown", "accident", "public_event", "water_logging",
                 "construction", "tree_fall", "procession", "protest", "vip_movement",
                 "congestion", "pot_holes", "road_conditions", "others", "Debris",
                 "Fog / Low Visibility"]
STATIONS      = [f"Station_{i}" for i in range(20)]


def _make_synthetic(n=200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    et  = rng.choice(EVENT_TYPES,  n)
    ec  = rng.choice(EVENT_CAUSES, n)
    ps  = rng.choice(STATIONS,     n)
    pri = rng.choice(["High", "Low"], n)
    rrc = rng.choice([True, False], n)
    return pd.DataFrame({"event_type": et, "event_cause": ec, "police_station": ps,
                         "priority": pri, "requires_road_closure": rrc})


def _load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(CSV_PATH, usecols=["event_type","event_cause",
                                             "police_station","priority",
                                             "requires_road_closure"])
        required = {"event_type","event_cause","police_station","priority","requires_road_closure"}
        if not required.issubset(df.columns):
            raise ValueError("Missing required columns")
        df = df.dropna(subset=list(required))
        if len(df) < 10:
            raise ValueError("Insufficient data rows")
        return df
    except Exception as e:
        print(f"[predictor] CSV load failed ({e}). Using synthetic data.")
        return _make_synthetic()


class EventPriorityPredictor:
    def __init__(self):
        df = _load_data()

        # --- Priority classifier ---
        df["target_priority"] = (df["priority"].str.strip().str.lower() == "high").astype(int)
        df["target_closure"]  = df["requires_road_closure"].astype(bool).astype(int)

        feat_cols = ["event_type", "event_cause", "police_station"]
        X = df[feat_cols].astype(str)
        y_pri  = df["target_priority"]
        y_clos = df["target_closure"]

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        ct  = ColumnTransformer([("ord", enc, feat_cols)], remainder="drop")

        self.pipe_priority = Pipeline([
            ("prep",  ct),
            ("clf",   RandomForestClassifier(n_estimators=150, max_depth=8,
                                              class_weight="balanced", random_state=42))
        ])
        self.pipe_closure = Pipeline([
            ("prep",  ct),
            ("clf",   RandomForestClassifier(n_estimators=150, max_depth=8,
                                              class_weight="balanced", random_state=99))
        ])
        self.pipe_priority.fit(X, y_pri)
        self.pipe_closure.fit(X, y_clos)

        self._known_stations = sorted(df["police_station"].dropna().unique().tolist())
        self._known_causes   = sorted(df["event_cause"].dropna().unique().tolist())
        self._known_types    = sorted(df["event_type"].dropna().unique().tolist())
        print("[predictor] Models trained successfully.")

    def predict(self, event_type: str, event_cause: str, station: str) -> dict:
        row = pd.DataFrame([{"event_type": str(event_type),
                              "event_cause": str(event_cause),
                              "police_station": str(station)}])
        pri_prob  = float(self.pipe_priority.predict_proba(row)[0][1])
        clos_prob = float(self.pipe_closure.predict_proba(row)[0][1])
        return {
            "priority_label":       "High" if pri_prob >= 0.5 else "Low",
            "priority_prob":        round(pri_prob, 4),
            "requires_closure":     clos_prob >= 0.5,
            "requires_closure_prob": round(clos_prob, 4),
        }

    @property
    def known_stations(self): return self._known_stations
    @property
    def known_causes(self):   return self._known_causes
    @property
    def known_types(self):    return self._known_types


# Module-level singleton (lazy)
_predictor_instance: EventPriorityPredictor = None


def get_predictor() -> EventPriorityPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = EventPriorityPredictor()
    return _predictor_instance


def predict_event_priority(event_type: str, event_cause: str, station: str) -> dict:
    """Public API: returns priority_prob and requires_closure_prob."""
    return get_predictor().predict(event_type, event_cause, station)


if __name__ == "__main__":
    result = predict_event_priority("unplanned", "accident", "Peenya")
    print(result)
