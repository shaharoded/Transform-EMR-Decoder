import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pickle
import numpy as np
from collections import Counter
from typing import Dict, Optional, List

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import *
from transform_emr.config.model_config import CHECKPOINT_PATH


class DataProcessor:
    """
    Handles the dataprocess needed to build the tokenizer / train / val / test.
    use max_input_days to trim a test dataset before using it for prediction.

    Expected columns for temporal_df: ['PatientID', 'ConceptName', 'StartDateTime', 'EndDateTime', 'Value']
    Expected columns for context_df: ['PatientID'] + context columns.

    Attributes:
    df (pd.DataFrame): Transformed long-format event dataframe after all processing.
    context_df (pd.DataFrame): Patient context dataframe with PatientID as index.
    scaler (StandardScaler): Scaler fitted to context_df and optionally saved to disk.
    checkpoint_path (str): Path to save the scaler / tokenizer at for later usage.

    """
    def __init__(self, df, context_df, 
                 tak_repo_path='transform_emr/config/tak_repo.pkl', 
                 max_input_days=None, 
                 scaler=None, 
                 checkpoint_path=CHECKPOINT_PATH):
        with open(tak_repo_path, "rb") as f:
            self.repo = pickle.load(f)

        if self.repo is None:
            raise ValueError("TAKRepository failed to load (None).")

        if not hasattr(self.repo, "get"):
            raise ValueError("TAKRepository must expose a .get(name) method.")
        
        # Handle compatability issue in complete flow
        if "StartTime" in df.columns and "EndTime" in df.columns:
            df.rename(columns={"StartTime": "StartDateTime", "EndTime": "EndDateTime"}, inplace=True)
        df['StartDateTime'] = pd.to_datetime(df['StartDateTime'], format='ISO8601', utc=True, errors='raise')
        df['StartDateTime'] = df['StartDateTime'].dt.tz_convert(None)
        df['EndDateTime'] = pd.to_datetime(df['EndDateTime'], format='ISO8601', utc=True, errors='raise')
        df['EndDateTime'] = df['EndDateTime'].dt.tz_convert(None)

        self.df = df.copy()
        self.context_df = context_df.copy()
        self.max_input_days = max_input_days
        self.scaler = scaler
        self.checkpoint_path = checkpoint_path


    def run(self):
        # Process on temporal_df
        self._validate_and_align_inputs()
        self._fix_back_to_back_intervals()
        self._truncate_after_terminal_event()
        self._normalize_time()
        self._expand_tokens() # Expands intervals into START/END tokens
        self._insert_null_tokens(gap_hrs=3)
        self._add_parent_raw_concepts()
        if self.max_input_days:
            self._cut_after_k_days()
        
        # Process on context_df
        if 'index' in self.context_df.columns:
            self.context_df = self.context_df.drop(columns=['index'])
        self.context_df = self.context_df.set_index("PatientID").drop(columns=["PatientID"], errors="ignore").astype("float32")
        self._fit_scaler()
        return self.df, self.context_df

    def _fit_scaler(self):
        """
        Fit and / or use a standard scaler on the context dataframe. 
        Will save the scaler in the checkpoints.
        """        
        if self.scaler is None:
            scaler = StandardScaler()
            self.context_df.loc[:, :] = scaler.fit_transform(self.context_df.values)
            os.makedirs(self.checkpoint_path, exist_ok=True)
            dump(scaler, os.path.join(self.checkpoint_path, 'scaler.pkl'))
        else:
            self.context_df.loc[:, :] = self.scaler.transform(self.context_df.values)

    def _validate_and_align_inputs(self):
        """
        Validates required columns, datetime types, and aligns PatientIDs between
        temporal (df) and context (patient_context_df) data. Will also sort the temporal data.

        Returns:
            Tuple of (cleaned_df, cleaned_patient_context_df)
        """
        # 1. Required columns check
        required_columns = ['PatientID', 'ConceptName', 'StartDateTime', 'EndDateTime', 'Value']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column in temporal data: {col}")
        if 'PatientID' not in self.context_df.columns:
            raise ValueError("Missing 'PatientID' column in context data")

        # 2. Check datetime dtypes
        if not pd.api.types.is_datetime64_any_dtype(self.df['StartDateTime']):
            raise TypeError("StartDateTime column must be of datetime64[ns] dtype.")
        if not pd.api.types.is_datetime64_any_dtype(self.df['EndDateTime']):
            raise TypeError("EndDateTime column must be of datetime64[ns] dtype.")

        # 3. Handle duplicate PatientIDs in context
        dupe_counts = self.context_df['PatientID'].value_counts()
        duplicates = dupe_counts[dupe_counts > 1]
        if not duplicates.empty:
            print(f"Found {len(duplicates)} PatientIDs with duplicate rows in context_df. Aggregating by max value...")
            self.context_df = self.context_df.groupby('PatientID').max().reset_index()

        # 4. Align temporal and context data
        temporal_ids = set(self.df['PatientID'])
        context_ids = set(self.context_df['PatientID'])

        missing_ids = temporal_ids - context_ids
        extra_ids = context_ids - temporal_ids

        if missing_ids:
            print(f"Adding {len(missing_ids)} missing PatientIDs to context_df with placeholder values (-1).")
            placeholder_df = pd.DataFrame({
                'PatientID': list(missing_ids),
                **{
                    col: [-1] * len(missing_ids)
                    for col in self.context_df.columns
                    if col != 'PatientID'
                }
            })
            self.context_df = pd.concat([self.context_df, placeholder_df], ignore_index=True)

        if extra_ids:
            print(f"Dropping {len(extra_ids)} unmatched PatientIDs from context_df.")
            self.context_df = self.context_df[self.context_df['PatientID'].isin(temporal_ids)].copy()

        # 5. Final integrity checks
        assert self.context_df['PatientID'].is_unique, "PatientID must be unique in context_df after alignment"
        assert set(self.df['PatientID']) == set(self.context_df['PatientID']), "Mismatched PatientIDs after alignment"
    

    def _fix_back_to_back_intervals(self, epsilon=pd.Timedelta(seconds=1)):
        """
        If an interval starts at exactly the same timestamp another one ends
        (same patient), shift the *start* forward by `epsilon` to preserve
        START/END ordering for tokenisation.
        """
        df = self.df.sort_values(['PatientID', 'StartDateTime']).reset_index(drop=True).copy()

        same_time = (
            (df['StartDateTime']
            == df.groupby('PatientID')['EndDateTime'].shift(1))
        )

        # shift only the conflicted rows
        df.loc[same_time, 'StartDateTime'] += epsilon
        # Also shift EndDateTime if duration would otherwise be negative/zero
        need_fix = df['EndDateTime'] <= df['StartDateTime']
        df.loc[need_fix, 'EndDateTime'] = df.loc[need_fix, 'StartDateTime'] + epsilon

        self.df = df


    def _truncate_after_terminal_event(self):
        """
        For each patient:
        - If a DEATH_TOKEN occurs within 30 days after a RELEASE_TOKEN:
            → Drop the DEATH_TOKEN.
            → Replace the RELEASE_TOKEN's ConceptName with DEATH_TOKEN (keep its time).
        - Then truncate any records after the first terminal event (RELEASE/DEATH).
        """
        def process_group(group):
            group = group.sort_values("StartDateTime").copy()

            # Handle RELEASE vs DEATH conflicts
            release_rows = group[group["ConceptName"] == RELEASE_TOKEN]
            death_rows = group[group["ConceptName"] == DEATH_TOKEN]

            if not release_rows.empty and not death_rows.empty:
                release_time = release_rows.iloc[0]["StartDateTime"]
                death_time = death_rows.iloc[0]["StartDateTime"]

                # If death is within 30 days after release → drop death, rename release
                if pd.Timedelta(0) <= (death_time - release_time) <= pd.Timedelta(days=30):
                    # Drop death row
                    group = group[group["ConceptName"] != DEATH_TOKEN]

                    # Replace concept name of release row
                    release_index = release_rows.index[0]
                    group.loc[release_index, "ConceptName"] = DEATH_TOKEN

            # Then truncate after first terminal event
            terminal_idx = group[group["ConceptName"].isin(TERMINAL_OUTCOMES)].index
            if not terminal_idx.empty:
                first_terminal_time = group.loc[terminal_idx[0], "StartDateTime"]
                group = group[group["StartDateTime"] <= first_terminal_time]

            return group

        self.df = (
            self.df.groupby("PatientID", group_keys=False)[self.df.columns]
                .apply(process_group)
                .reset_index(drop=True)
        )
    

    def _normalize_time(self):
        df = self.df.copy()
        df["IsAdmission"] = df["ConceptName"] == ADMISSION_TOKEN
        df["VisitCounter"] = df.groupby("PatientID")["IsAdmission"].cumsum()
        df["VisitID"] = df["PatientID"].astype(str) + "_" + df["VisitCounter"].astype(str)
        df["VisitStart"] = df.groupby("VisitID")["StartDateTime"].transform('min')
        df["RelStartTime"] = (df["StartDateTime"] - df["VisitStart"]).dt.total_seconds() / 3600.0 # In hours
        df["RelEndTime"] = (df["EndDateTime"] - df["VisitStart"]).dt.total_seconds() / 3600.0 # In hours
        self.df = df


    def _add_parent_raw_concepts(self):
        """
        Adds a 'ParentRawConcepts' column to self.df, which contains a list of
        top-level raw concepts for each ConceptName based on the TAKRepository.
        """
        def __deps_from_derived_from(tak_obj):
            """
            Get the list of TAK names that the given TAK object is derived from.
            """
            if not hasattr(tak_obj, "derived_from"):
                raise ValueError(f"TAK '{tak_obj.name}' has no 'derived_from' attribute.")

            df = tak_obj.derived_from

            if df is None:
                return []

            if isinstance(df, str):
                return [df]

            if isinstance(df, list):
                out = []
                for item in df:
                    if isinstance(item, dict) and "name" in item:
                        out.append(item["name"])
                    elif isinstance(item, str):
                        out.append(item)
                    else:
                        raise ValueError(
                            f"Invalid derived_from entry in TAK '{tak_obj.name}': {item}"
                        )
                return out

            raise ValueError(
                f"Invalid derived_from type for TAK '{tak_obj.name}': {type(df)}"
            )

        def __resolve_top_raw(name: str, seen=None) -> str:
            """
            Recursively resolves the top-level raw concept for a given TAK name.
            Raises an error if a cycle is detected or if the TAK is not found.
            """
            if seen is None:
                seen = set()

            if name in seen:
                raise ValueError(f"Cycle detected while resolving raw parent for '{name}'.")

            seen.add(name)

            tak = self.repo.get(name)
            if tak is None:
                raise ValueError(f"Concept '{name}' not found in TAKRepository.")

            if getattr(tak, "family", None) == "raw-concept":
                return tak.name

            deps = __deps_from_derived_from(tak)
            if not deps:
                raise ValueError(
                    f"TAK '{name}' is not raw-concept and has empty derived_from."
                )

            if len(deps) > 1:
                # This should only happen for patterns, but still supported
                raise ValueError(
                    f"Non-pattern TAK '{name}' has multiple derived_from parents: {deps}"
                )

            return __resolve_top_raw(deps[0], seen)
        

        def _parents_for_concept(concept_name: str):
            if concept_name in ("[NULL]", "[CTX]", "[MASK]", "[PAD]"):
                return ["[NULL]"]
            tak = self.repo.get(concept_name)
            if tak is None:
                raise ValueError(f"Concept '{concept_name}' not found in TAKRepository.")

            deps = __deps_from_derived_from(tak)
            if not deps:
                # raw concepts must have empty derived_from
                if getattr(tak, "family", None) != "raw-concept":
                    raise ValueError(
                        f"TAK '{concept_name}' has no derived_from but is not raw-concept."
                    )
                return [tak.name]

            parents = set()
            for dep in deps:
                parents.add(__resolve_top_raw(dep))

            return sorted(parents)
        
        self.df['ParentRawConcepts'] = self.df['Concept'].apply(_parents_for_concept)

    
    def _expand_tokens(self, min_interval_duration_sec=1):
        """
        Expands events into tokens with timepoints.

        - Splits state events into START and END tokens.
        - Keeps instantaneous events as single tokens.
        
        Returns:
            DataFrame with ['PatientID', 'RawConcept', 'Concept', 'ValueToken', 'PositionToken', 'TimePoint'].
        """
        df = self.df
        rows = []
        for row in df.itertuples(index=False):
            duration_sec = (row.EndDateTime - row.StartDateTime).total_seconds()
            base_token = f"{row.ConceptName}_{row.Value}" if row.Value not in ("True", "TRUE") else row.ConceptName
            concept = row.ConceptName
            value = base_token

            # Create interval tokenization if duration > threshold
            is_interval = (duration_sec > min_interval_duration_sec)
            pos_tokens = []

            if is_interval:
                pos_tokens = ["START", "END"]
                time_points = [row.RelStartTime, row.RelEndTime]
            else:
                pos_tokens = [""]
                time_points = [row.RelStartTime]

            for pos, tp in zip(pos_tokens, time_points):
                full_token = f"{base_token}_{pos}" if pos else base_token
                rows.append({
                    'PatientID': row.PatientID,
                    'Concept': concept,
                    'ValueToken': value,
                    'PositionToken': full_token,
                    'TimePoint': tp
                })

        df = pd.DataFrame(rows)

        # --- Sort and compute time deltas ---
        self.df = df.sort_values(['PatientID', 'TimePoint'])
    

    def _insert_null_tokens(self, gap_hrs: int = 3) -> None:
        """
        Insert a single synthetic [NULL] token whenever there is a gap > `gap_hrs`
        *and* no interval is open (open_stack==0).  Token is placed at gap midpoint.
        """
        if gap_hrs <= 0:
            return

        rows_out = []
        for pid, grp in self.df.groupby("PatientID"):
            grp = grp.sort_values("TimePoint")            # safety
            open_stack, last_tp = 0, None

            for row in grp.itertuples(index=False):
                tp = row.TimePoint

                # ---------- gap check ----------
                if last_tp is not None:
                    gap = tp - last_tp
                    if gap >= gap_hrs and open_stack == 0:
                        rows_out.append({
                            "PatientID": pid,
                            "RawConcept": "[NULL]",
                            "Concept":    "[NULL]",
                            "ValueToken": "[NULL]",
                            "PositionToken": "[NULL]",
                            "TimePoint":  last_tp + gap / 2
                        })

                # ---------- interval stack ----------
                tok = row.PositionToken
                if tok.endswith("_START"):
                    open_stack += 1
                elif tok.endswith("_END"):
                    open_stack = max(0, open_stack - 1)

                # ---------- keep real event ----------
                rows_out.append(row._asdict())
                last_tp = tp

        # Replace dataframe
        self.df = pd.DataFrame(rows_out)


    def _cut_after_k_days(self):
        """
        Trims token-level data to only include tokens occurring within the first `k` days (from admission).
        Drops visits where no events remain after truncation.

        NOTE: This version fits a data process where PatientID is actually the visitID, meaning every ID belongs 
        to only 1 group of records. If you want generation based on PatientID that can have the information of a few 
        visits you'll need to change the key here to VisitCounter, but to ensure it is also passed from _expand_tokens().

        """
        df = self.df
        k_hours = self.max_input_days * 24

        # Only keep visits that originally last longer than k days
        visit_max_times = df.groupby("PatientID")["TimePoint"].max()
        long_enough_visits = visit_max_times[visit_max_times > k_hours].index
        df = df[df["PatientID"].isin(long_enough_visits)].copy()

        # Keep only events up to k_days
        df = df[df["TimePoint"] <= k_hours].copy()

        # Drop visits with no records remaining
        remaining_visits = df.groupby("PatientID").size()
        df = df[df["PatientID"].isin(remaining_visits[remaining_visits > 1].index)]

        self.df = df


class EMRTokenizer:
    """
    A custom tokenizer objest to match this model's requirement.
    build this object with your full training data to ensure it builds properly.
    Token weights are determined by token frequency in cohort.

    Attributes:
        token2id (Dict[str, int]): Full vocabulary mapping ("GLUCOSE_STATE_HIGH_START").
        id2token (Dict[int, str]): Reverse mapping for decoding.
        rawconcept2id (Dict[str, int]): Vocabulary mapping for raw concepts only ("GLUCOSE").
        concept2id (Dict[str, int]): Vocabulary mapping for concepts only ("GLUCOSE_STATE"/ "GLUCOSE_TREND")..
        value2id (Dict[str, int]): Vocabulary mapping for concepts+values ("GLUCOSE_STATE_HIGH")
        special_tokens (List[str]): Special tokens like ["MASK"].
        token_weights (torch.Tensor): Weights used in loss to emphasize important tokens.
        important_token_ids (torch.Tensor): Token IDs with weight > 1.0.
        token_counts (torch.Tensor): Token counts (distribution).
        pad_token_id (int): ID for padding token.
        mask_token_id (int): ID for MASK token.
        ctx_token_id (int): ID for context token.
        null_token_id (int): ID for NULL token.
    """
    def __init__(self, token2id, rawconcept2id, concept2id, value2id, special_tokens, 
                 token_weights, important_token_ids, token_counts,
                 tokenid2parent_raw_ids, parent_pad_len):
        self.token2id = token2id
        self.id2token = {i: tok for tok, i in token2id.items()}
        self.rawconcept2id = rawconcept2id
        self.concept2id = concept2id
        self.value2id = value2id
        self.special_tokens = special_tokens
        self.token_weights = token_weights
        self.important_token_ids = important_token_ids
        self.token_counts = token_counts
        self.tokenid2parent_raw_ids = tokenid2parent_raw_ids
        self.parent_pad_len = parent_pad_len
        
        # Validate presence of mandatory special tokens
        required_specials = ["[PAD]", "[MASK]", "[NULL]", "[CTX]"]
        for tok in required_specials:
            if tok not in token2id:
                raise ValueError(f"[Tokenizer Error] Missing required special token: {tok}")

        self.pad_token_id = token2id["[PAD]"]
        self.mask_token_id = token2id["[MASK]"]
        self.null_token_id = token2id["[NULL]"]
        self.ctx_token_id = token2id["[CTX]"]


    @classmethod
    def from_processed_df(cls, df, special_tokens=["[PAD]", "[MASK]", "[NULL]", "[CTX]"]):
        """
        Takes in a processed dataframe from DataProcessor.run() and builds the 
        tokenizer vocabularies and weights.
        Args:
            df (pd.DataFrame): Processed dataframe with columns ['ParentRawConcepts', 'Concept', 'ValueToken', 'PositionToken'].
            special_tokens (List[str]): List of special tokens to include in the vocab.
        Returns:
            EMRTokenizer: Initialized tokenizer object.
        """
        # --- raw concepts are taken from ParentRawConcepts (list[str]) ---
        if "ParentRawConcepts" not in df.columns:
            raise ValueError("ParentRawConcepts column missing. Dataset not preprocessed.")

        raw_set = set()
        for lst in df["ParentRawConcepts"]:
            if not isinstance(lst, list):
                raise ValueError(f"ParentRawConcepts must be list[str], got {type(lst)}")
            for x in lst:
                raw_set.add(x)

        raw_concepts = sorted(raw_set)
        concepts = sorted(df['Concept'].unique())
        values = sorted(df['ValueToken'].unique())
        positions = sorted(df['PositionToken'].unique())

        # Sort and inject special tokens in order
        raw_concepts = [tok for tok in raw_concepts if tok not in special_tokens]
        concepts = [tok for tok in concepts if tok not in special_tokens]
        values = [tok for tok in values if tok not in special_tokens]
        positions = [tok for tok in positions if tok not in special_tokens]

        raw_concepts = special_tokens + raw_concepts
        concepts = special_tokens + concepts
        values = special_tokens + values
        positions = special_tokens + positions

        token2id = {tok: i for i, tok in enumerate(positions)}
        rawconcept2id = {tok: i for i, tok in enumerate(raw_concepts)}
        concept2id = {tok: i for i, tok in enumerate(concepts)}
        value2id = {tok: i for i, tok in enumerate(values)}

        # === Define Token Weights ===
        # Only used for regular BCELoss (Embedder), not Focal (Transformer)
        # Initialize weights (data driven)
        token_weights = torch.ones(len(token2id))

        for outcome in OUTCOMES:
            tok_id = token2id.get(outcome)
            if tok_id is not None:
                token_weights[tok_id] = 10.0
        for term in TERMINAL_OUTCOMES:
            tok_id = token2id.get(term)
            if tok_id is not None:
                token_weights[tok_id] = 15.0
        for ignore_tok in special_tokens + [ADMISSION_TOKEN]:
            tok_id = token2id.get(ignore_tok)
            if tok_id is not None:
                token_weights[tok_id] = 0.0
        
        important_token_ids = (token_weights > 1.0).nonzero(as_tuple=True)[0]

        # Add token distribution
        count_series = df["PositionToken"].value_counts()
        counts_vec   = torch.zeros(len(token2id), dtype=torch.long)
        for tok, cnt in count_series.items():
            counts_vec[token2id[tok]] = cnt
        
        # ---- Build tokenid2parent_raw_ids LUT for inference ----
        # Expect df has PositionToken + ParentRawConcepts
        pos_to_parents = {}

        for pos_tok, grp in df.groupby("PositionToken"):
            # enforce determinism: all rows for same PositionToken must agree
            unique_lists = grp["ParentRawConcepts"].apply(lambda x: tuple(x)).unique()
            if len(unique_lists) != 1:
                raise ValueError(
                    f"[Tokenizer Error] Inconsistent ParentRawConcepts for PositionToken='{pos_tok}'. "
                    f"Got {len(unique_lists)} variants (example={unique_lists[:3]})."
                )
            pos_to_parents[pos_tok] = list(unique_lists[0])

        # compute global Pmax
        Pmax = max((len(v) for v in pos_to_parents.values()), default=1)
        if Pmax <= 0:
            raise ValueError("[Tokenizer Error] Computed Pmax <= 0 for parent raw concepts.")

        # fill LUT with PAD id from token2id
        pad_id = token2id["[PAD]"]
        lut = torch.full((len(token2id), Pmax), pad_id, dtype=torch.long)

        # helper for encoding parent names -> ids
        def encode_parent_names(parent_names):
            ids = []
            for name in parent_names:
                if name not in rawconcept2id:
                    raise ValueError(f"[Tokenizer Error] Raw parent '{name}' missing from rawconcept2id.")
                ids.append(rawconcept2id[name])
            if len(ids) == 0:
                raise ValueError("[Tokenizer Error] Empty parent list encountered.")
            return ids

        # populate LUT by token id
        for tok, tid in token2id.items():
            # Special tokens map to [NULL] as their parent raw concept
            if tok in special_tokens:
                if "[NULL]" not in rawconcept2id:
                    raise ValueError("[Tokenizer Error] '[NULL]' missing from rawconcept2id.")
                null_id = rawconcept2id["[NULL]"]
                lut[tid, 0] = null_id
                continue
            
            if tok not in pos_to_parents:
                raise ValueError(
                    f"[Tokenizer Error] Missing ParentRawConcepts for token '{tok}'. "
                    f"Processed df must include all PositionToken values."
                )
            ids = encode_parent_names(pos_to_parents[tok])
            lut[tid, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        return cls(token2id, rawconcept2id, concept2id, value2id,special_tokens, token_weights, 
                   important_token_ids, counts_vec, lut, Pmax)

    def save(self, path=os.path.join(CHECKPOINT_PATH, 'tokenizer.pt')):
        torch.save({
            'token2id': self.token2id,
            'rawconcept2id': self.rawconcept2id,
            'concept2id': self.concept2id,
            'value2id': self.value2id,
            'special_tokens': self.special_tokens,
            'token_weights': self.token_weights,
            'important_token_ids': self.important_token_ids,
            'token_counts': self.token_counts,
            'tokenid2parent_raw_ids': self.tokenid2parent_raw_ids,
            'parent_pad_len': self.parent_pad_len,
            'fingerprint': self.fingerprint()
        }, path)


    @classmethod
    def load(cls, path=os.path.join(CHECKPOINT_PATH, 'tokenizer.pt')):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj = torch.load(path, map_location=device)

        tokenizer = cls(
            token2id=obj['token2id'],
            rawconcept2id=obj['rawconcept2id'],
            concept2id=obj['concept2id'],
            value2id=obj['value2id'],
            special_tokens=obj['special_tokens'],
            token_weights=obj['token_weights'].to(device),
            important_token_ids=obj['important_token_ids'].to(device),
            token_counts=obj['token_counts'].to(device),
            tokenid2parent_raw_ids=obj['tokenid2parent_raw_ids'].to(device),
            parent_pad_len=obj['parent_pad_len'],
        )
        tokenizer._loaded_fingerprint = obj.get('fingerprint')
        return tokenizer
    

    def fingerprint(self):
        return hash(frozenset(self.token2id.items()))


class EMRDataset(Dataset):
    def __init__(self, processed_df: pd.DataFrame, context_df: pd.DataFrame, tokenizer: EMRTokenizer):
        """
        processed_df: processed DataFrame after running DataProcessor.run() on the original temporal df.
        context_df: Also processed by DataProcessor.run().

        This class performs data cleaning, as well as prperation of data for input as train of for inference as test.

        Attr:
            self.tokenizer (EMRTokenizer): A tokenizer object capable of encoding and decoding all temporal tokens (and subtokens as required)
            self.context_df (pd.DataFrame): Patient-level context features (indexed by PatientID), scaled to zero mean and unit variance.
            self.tokens_df (pd.DataFrame): Long-format temporal event dataframe with per-token attributes and timing features.
            self.patient_ids (np.ndarray): Array of unique PatientIDs present in the dataset.
            self.patient_groups (Dict[str, pd.DataFrame]): Mapping from PatientID to their corresponding token DataFrame.
        """
        self.tokenizer = tokenizer
        self.tokens_df = processed_df
        self.context_df = context_df

        # --- Mapping function with warnings ---
        def safe_map(column, vocab, label):
            mapped = self.tokens_df[column].map(vocab)
            unknown = self.tokens_df.loc[mapped.isna(), column].unique()
            if len(unknown) > 0:
                print(f"[Warning][EMRDataset] Unknown {label} values found (count={len(unknown)}):")
                for tok in unknown[:10]:  # only print a sample
                    print(f"  - {tok}")
                raise ValueError(f"[Dataset Error] Found unknown {label} entries. Tokenizer or parsing may be out of sync.")
            return mapped.astype(int)

        # --- Map with validation ---
        self.tokens_df['ConceptID']    = safe_map('Concept', self.tokenizer.concept2id, 'Concept')
        self.tokens_df['ValueID']      = safe_map('ValueToken', self.tokenizer.value2id, 'ValueToken')
        self.tokens_df['PositionID']   = safe_map('PositionToken', self.tokenizer.token2id, 'PositionToken')

        self.tokens_df["ParentRawConceptIDs"] = self.tokens_df["ParentRawConcepts"].apply(
            self.__encode_parent_list
        )

        self.patient_ids = self.tokens_df['PatientID'].unique()
        self.patient_groups = {pid: self.tokens_df[self.tokens_df['PatientID'] == pid] for pid in self.patient_ids}

    def __encode_parent_list(self, parents: List[str]) -> List[int]:
        ids = []
        for p in parents:
            if p not in self.tokenizer.rawconcept2id:
                raise ValueError(f"Raw parent concept '{p}' missing from tokenizer vocabulary.")
            ids.append(self.tokenizer.rawconcept2id[p])
        return ids


    def __len__(self):
        """
        Returns the number of patients in the dataset
        """
        return len(self.patient_ids)


    def __getitem__(self, idx, allow_debug=False):
        """
        Returns the subset of records for 1 patient.
        """
        pid = self.patient_ids[idx]
        df = self.patient_groups[pid]

        # ---------------------------
        # Build padded parent_raw_ids: [T, P]
        # ---------------------------
        if "ParentRawConceptIDs" not in df.columns:
            raise ValueError(
                f"[Dataset Error] Missing ParentRawConceptIDs for patient {pid}. "
                f"Make sure EMRDataset built ParentRawConcepts/IDs before grouping."
            )

        parent_lists = df["ParentRawConceptIDs"].tolist()

        if len(parent_lists) == 0:
            # extremely rare, but keep it safe
            max_p = 1
        else:
            # validate type and compute max parents for this patient
            for i, lst in enumerate(parent_lists[:10]):  # sample validation
                if not isinstance(lst, list):
                    raise ValueError(
                        f"[Dataset Error] ParentRawConceptIDs must be list[int]. "
                        f"Got {type(lst)} at row {i} for patient {pid}."
                    )
            max_p = max(len(x) for x in parent_lists) if parent_lists else 1
            if max_p == 0:
                raise ValueError(
                    f"[Dataset Error] Patient {pid} has empty ParentRawConceptIDs lists."
                )

        parent_raw_ids = torch.full(
            (len(parent_lists), max_p),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )

        for i, lst in enumerate(parent_lists):
            if len(lst) == 0:
                raise ValueError(
                    f"[Dataset Error] Empty ParentRawConceptIDs for patient {pid} at token index {i}."
                )
            parent_raw_ids[i, :len(lst)] = torch.tensor(lst, dtype=torch.long)

        if allow_debug:
            # Check each ID column (no RawConceptID anymore)
            for col_name, vocab_dict in [
                ("ConceptID", self.tokenizer.concept2id),
                ("ValueID", self.tokenizer.value2id),
                ("PositionID", self.tokenizer.token2id),
            ]:
                ids = df[col_name].values
                max_valid = len(vocab_dict) - 1

                if (ids > max_valid).any():
                    bad_ids = ids[ids > max_valid]
                    print(f"ERROR: Patient {pid} has out-of-bounds {col_name}: {bad_ids}")
                    print(f"  Valid range for {col_name}: [0, {max_valid}]")
                    print(f"  Actual range: [{ids.min()}, {ids.max()}]")

                    bad_positions = np.where(ids > max_valid)[0]
                    for pos in bad_positions[:3]:
                        print(f"  Position {pos}: {col_name}={ids[pos]}")
                        if col_name == "PositionID":
                            original_token = df.iloc[pos]["PositionToken"]
                            print(f"    Original token: '{original_token}'")

                    raise ValueError(f"Found out-of-bounds {col_name} for patient {pid}")

            # Validate parent_raw_ids bounds too
            max_raw_valid = len(self.tokenizer.rawconcept2id) - 1
            if (parent_raw_ids > max_raw_valid).any():
                bad = parent_raw_ids[parent_raw_ids > max_raw_valid].unique()
                raise ValueError(
                    f"Found out-of-bounds parent_raw_ids for patient {pid}: {bad.tolist()} "
                    f"(valid range [0, {max_raw_valid}])"
                )

        return {
            "parent_raw_ids": parent_raw_ids,  # [T, P]
            "concept_ids": torch.tensor(df["ConceptID"].values, dtype=torch.long),
            "value_ids": torch.tensor(df["ValueID"].values, dtype=torch.long),
            "position_ids": torch.tensor(df["PositionID"].values, dtype=torch.long),
            "abs_ts": torch.tensor(df["TimePoint"].values, dtype=torch.float32) / 336.0,  # [0,1]
            "context_vec": torch.tensor(self.context_df.loc[pid].values, dtype=torch.float32),
            "targets": torch.tensor(df["PositionID"].values, dtype=torch.long),
        }


def collate_emr(batch, pad_token_id=0):
    """
    Collates a batch of patient EMR sequences into padded tensors.

    Each sequence contains:
        - Parents Concept ID (2D: [T, P])
        - Concept ID (1D: [T])
        - Value ID (1D: [T])
        - Position ID (1D: [T]) (used for prediction)
        - Absolute time (Δt since admission) (1D: [T])
        - Patient context vector (no padding) (1D: [C])
    Returns:
        Dictionary of padded tensors: [B, T_max] + context_vec [B, C]
    
    NOTE: Padding token id ([PAD]) is always 0. Time Padding should be 0.0.
    """
    batch_size = len(batch)
    max_len = max(x["position_ids"].shape[0] for x in batch)
    max_p   = max(x["parent_raw_ids"].shape[1] for x in batch)
    
    def pad_2d(seq2d, pad_val=pad_token_id, dtype=torch.long):
        out = torch.full((batch_size, max_len, max_p), pad_val, dtype=dtype)
        for i, s in enumerate(seq2d):
            t, p = s.shape
            out[i, :t, :p] = s
        return out

    def pad_1d(seq1d, pad_val=pad_token_id, dtype=torch.long):
        out = torch.full((batch_size, max_len), pad_val, dtype=dtype)
        for i, s in enumerate(seq1d):
            out[i, :len(s)] = s
        return out

    parent_raw_ids = pad_2d([x["parent_raw_ids"] for x in batch], pad_val=pad_token_id)
    concept_ids    = pad_1d([x["concept_ids"] for x in batch], pad_val=pad_token_id)
    value_ids      = pad_1d([x["value_ids"] for x in batch], pad_val=pad_token_id)
    position_ids   = pad_1d([x["position_ids"] for x in batch], pad_val=pad_token_id)
    abs_ts         = pad_1d([x["abs_ts"] for x in batch], pad_val=0.0, dtype=torch.float32)
    context_vecs   = torch.stack([x["context_vec"] for x in batch])

    return {
        "parent_raw_ids": parent_raw_ids,
        "concept_ids": concept_ids,
        "value_ids": value_ids,
        "position_ids": position_ids,
        "abs_ts": abs_ts,
        "context_vec": context_vecs,
        "targets": position_ids.clone(),
    }


def get_dataloader(
        dataset,
        batch_size: int,
        collate_fn,
        *,
        oversample: bool = False,
        num_workers: int = os.cpu_count(),
        replacement: bool = True,
        class_weights: Optional[Dict] = None,
):
    """
    Build a DataLoader for an EMRDataset.
    If `oversample=True`, uses a WeightedRandomSampler to balance terminal outcomes.
    """

    # ---------- helper ----------
    def _label_visit(token_df, tokenizer):
        """
        Return an int label per patient trajectory:

            0 = DEATH present
            1 = ≥1 COMPLICATION present (no DEATH)
            2 = RELEASE present only
            3 = OTHER / still-in-hospital (no terminal/outcome token)

        These labels feed a WeightedRandomSampler so that DEATH and COMPLICATION
        patients are drawn more often without duplicating rows.
        """
        ids = set(token_df["PositionID"].tolist())
        tid = tokenizer.token2id

        if tid[DEATH_TOKEN] in ids:
            return 0
        if any(tid.get(c) in ids for c in OUTCOMES if c in tid):
            return 1 
        if tid[RELEASE_TOKEN] in ids:
            return 2
        return 3          # no terminal/outcome yet

    # ---------- no oversampling ----------
    if not oversample:
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          pin_memory=True)

    # ---------- build sample weights ----------
    labels = [_label_visit(dataset.patient_groups[pid], dataset.tokenizer)
              for pid in dataset.patient_ids]

    if class_weights is None:
        counts = Counter(labels)
        class_weights = {c: 1.0 / max(1, n) for c, n in counts.items()}

    sample_w = torch.tensor([class_weights[l] for l in labels],
                            dtype=torch.float32)

    sampler = WeightedRandomSampler(sample_w,
                                    num_samples=len(sample_w),
                                    replacement=replacement)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      sampler=sampler,
                      collate_fn=collate_fn,
                      num_workers=num_workers,
                      pin_memory=True)