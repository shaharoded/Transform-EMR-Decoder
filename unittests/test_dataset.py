from torch.utils.data import DataLoader
import os
from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset, collate_emr
import pandas as pd
import pytest
import pickle
from transform_emr.dataset import DataProcessor
from transform_emr.config.dataset_config import RELEASE_TOKEN, DEATH_TOKEN, ADMISSION_TOKEN

# --- Mock Setup (Reuse or place in conftest.py) ---
class MockTAK:
    def __init__(self, name, derived_from=None, family=None):
        self.name = name
        self.derived_from = derived_from
        self.family = family

class MockRepo:
    def __init__(self, items):
        self.items = items
    def get(self, name):
        return self.items.get(name)

@pytest.fixture
def mock_tak_repo(tmp_path):
    repo = MockRepo({
        "A": MockTAK("A", family="raw-concept"),
        "B": MockTAK("B", derived_from=["A"]),
        "C": MockTAK("C", derived_from=["B"]),          # Nested derivation
        "D": MockTAK("D", family="raw-concept"),        # Independent
        "Pattern": MockTAK("Pattern", derived_from=["A", "D"]), # Multi-parent
        "[NULL]": MockTAK("[NULL]", family="raw-concept"),
        RELEASE_TOKEN: MockTAK(RELEASE_TOKEN, family="raw-concept"),
        DEATH_TOKEN: MockTAK(DEATH_TOKEN, family="raw-concept"),
        ADMISSION_TOKEN: MockTAK(ADMISSION_TOKEN, family="raw-concept"),
    })
    repo_path = tmp_path / "tak_repo.pkl"
    with open(repo_path, "wb") as f:
        pickle.dump(repo, f)
    return str(repo_path)

@pytest.fixture
def base_ctx():
    return pd.DataFrame({'PatientID': [1], 'Age': [30]})

@pytest.mark.order(1)
def test_synthetic_data_pipeline(tmp_path, capsys, mock_tak_repo):
    # --- create a tiny two‑patient temporal table ---
    df = pd.DataFrame({
        'PatientID':     [1, 1, 2, 2],
        'StartDateTime': ['2020-01-01 00:00:00', '2020-01-02 01:00:00',
                          '2020-02-01 00:00:00', '2020-02-02 02:00:00'],
        'EndDateTime':   ['2020-01-01 03:00:00', '2020-01-02 03:00:00',
                          '2020-02-01 04:00:00', '2020-02-02 07:00:00'],
        'ConceptName':   ['A', 'B', 'A', 'B'],
        'Value':         [1.0, 2.0, 3.0, 4.0]
    })
    ctx = pd.DataFrame({
        'PatientID': [1, 2],
        'Gender':    [0, 1],
        'Age':       [30, 40]
    })

    # run preprocessing with and without max_input_days
    # Pass mock repo and tmp_path for checkpoints
    proc_t = DataProcessor(df, ctx, tak_repo_path=mock_tak_repo, max_input_days=1, checkpoint_path=str(tmp_path))
    temporal_df_t, _ = proc_t.run()
    
    proc = DataProcessor(df, ctx, tak_repo_path=mock_tak_repo, checkpoint_path=str(tmp_path))
    temporal_df, context_df = proc.run()

    # lengths should differ when max_input_days=1
    assert len(temporal_df_t) != len(temporal_df)

    # print dataframes for debugging
    print("\n--- temporal_df (full) ---")
    print(temporal_df)
    print("\n--- temporal_df_t (truncated) ---")
    print(temporal_df_t)
    print("\n--- context_df ---")
    print(context_df)

    # now build tokenizer & dataset on full temporal_df
    tokenizer = EMRTokenizer.from_processed_df(temporal_df)
    tokenizer.save(os.path.join(tmp_path, 'tokenizer.pt'))
    ds = EMRDataset(temporal_df, context_df, tokenizer=tokenizer)

    # collate one batch of size 2
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_emr)
    batch = next(iter(dl))

    # expecting these tensors in batch and correct batch dimension
    expected_keys = (
        'parent_raw_ids', 'concept_ids', 'value_ids',
        'position_ids', 'abs_ts', 'context_vec', 'targets'
    )
    for key in expected_keys:
        assert key in batch, f"Missing {key} in batch"
    assert batch['position_ids'].shape[0] == 2
    # === Test NULL token insertion ===
    # There should be at least one [NULL] row with intermediate TimePoint
    null_rows = temporal_df[temporal_df['Concept'] == '[NULL]']
    assert not null_rows.empty, 'Expected at least one NULL token row'
    # Pick first NULL and verify its TimePoint is midpoint of its neighbors
    pid = null_rows.iloc[0]['PatientID']
    df_pid = temporal_df[temporal_df['PatientID'] == pid].sort_values('TimePoint').reset_index(drop=True)
    # locate NULL index
    null_idx = df_pid.index[df_pid['Concept'] == '[NULL]'][0]
    # ensure it's not at the very start or end
    assert 0 < null_idx < len(df_pid) - 1
    t_prev = df_pid.loc[null_idx - 1, 'TimePoint']
    t_next = df_pid.loc[null_idx + 1, 'TimePoint']
    t_null = df_pid.loc[null_idx, 'TimePoint']
    # midpoint check
    expected_mid = t_prev + (t_next - t_prev) / 2
    assert pytest.approx(expected_mid, rel=1e-3) == t_null

@pytest.mark.order(1)
def test_fix_back_to_back_intervals(tmp_path, mock_tak_repo):  # <--- Added mock_tak_repo fixture here
    # Setup: Patient 1 has an interval ending exactly when the next one starts
    df = pd.DataFrame({
        'PatientID': [1, 1],
        'ConceptName': ['A', 'A'],
        'StartDateTime': [
            pd.Timestamp('2020-01-01 10:00:00'),
            pd.Timestamp('2020-01-01 11:00:00') # Starts exactly when prev ends
        ],
        'EndDateTime': [
            pd.Timestamp('2020-01-01 11:00:00'), # Ends at 11:00:00
            pd.Timestamp('2020-01-01 12:00:00')
        ],
        'Value': [1, 2]
    })
    ctx = pd.DataFrame({'PatientID': [1], 'Age': [30]})
    
    # Run processor
    # Pass the mock_tak_repo path instead of "dummy_path"
    processor = DataProcessor(df, ctx, tak_repo_path=mock_tak_repo, checkpoint_path=str(tmp_path))
    
    # Run the specific method we are testing
    processor._fix_back_to_back_intervals() 
    
    res = processor.df
    
    # Assertions
    # The first event should remain untouched
    assert res.iloc[0]['EndDateTime'] == pd.Timestamp('2020-01-01 11:00:00')
    
    # The second event start should be shifted by epsilon (1 second)
    expected_start = pd.Timestamp('2020-01-01 11:00:01')
    assert res.iloc[1]['StartDateTime'] == expected_start
    
    # The duration check: EndDateTime should typically not shift unless start > end
    assert res.iloc[1]['EndDateTime'] == pd.Timestamp('2020-01-01 12:00:00')


# --- Unit Tests ---
@pytest.mark.order(1)
def test_truncate_after_terminal_event(tmp_path, mock_tak_repo, base_ctx):
    """
    Test 2 behaviors:
    1. Truncate all events after the first RELEASE or DEATH.
    2. If RELEASE is followed by DEATH within 30 days, merge them (RELEASE becomes DEATH).
    """
    df = pd.DataFrame({
        'PatientID': [1, 1, 1, 1],
        'ConceptName': ['A', RELEASE_TOKEN, DEATH_TOKEN, 'B'],
        'StartDateTime': [
            pd.Timestamp('2020-01-01 10:00'),
            pd.Timestamp('2020-01-01 12:00'),
            pd.Timestamp('2020-01-01 13:00'), # 1 hour after release (within 30 days)
            pd.Timestamp('2020-01-02 10:00')  # Event after terminal -> Should be dropped
        ],
        'EndDateTime': [
            pd.Timestamp('2020-01-01 11:00'),
            pd.Timestamp('2020-01-01 12:00'),
            pd.Timestamp('2020-01-01 13:00'),
            pd.Timestamp('2020-01-02 11:00')
        ],
        'Value': [1, 1, 1, 1]
    })
    
    proc = DataProcessor(df, base_ctx, tak_repo_path=mock_tak_repo, checkpoint_path=str(tmp_path))
    proc._truncate_after_terminal_event()
    res = proc.df

    # 1. 'B' (after terminal) should be dropped
    assert 'B' not in res['ConceptName'].values
    
    # 2. RELEASE row should be renamed to DEATH
    assert res.iloc[1]['ConceptName'] == DEATH_TOKEN
    
    # 3. The original DEATH row (row index 2) is dropped (merged into index 1)
    assert len(res) == 2 
    
    # 4. Check timestamps of the merged event (should keep RELEASE time)
    assert res.iloc[1]['StartDateTime'] == pd.Timestamp('2020-01-01 12:00')

@pytest.mark.order(1)
def test_normalize_time(tmp_path, mock_tak_repo, base_ctx):
    """
    Test normalization relative to the 'ADMISSION_TOKEN' start time.
    """
    df = pd.DataFrame({
        'PatientID': [1, 1, 1],
        'ConceptName': [ADMISSION_TOKEN, 'A', 'B'],
        'StartDateTime': [
            pd.Timestamp('2020-01-01 10:00'), # Visit Start
            pd.Timestamp('2020-01-01 12:30'), # +2.5 hours
            pd.Timestamp('2020-01-02 10:00'), # +24 hours
        ],
        'EndDateTime': [
             pd.Timestamp('2020-01-01 10:00'),
             pd.Timestamp('2020-01-01 13:30'),
             pd.Timestamp('2020-01-02 11:00'),
        ],
        'Value': [1, 1, 1]
    })

    proc = DataProcessor(df, base_ctx, tak_repo_path=mock_tak_repo, checkpoint_path=str(tmp_path))
    proc._normalize_time()
    res = proc.df

    assert 'RelStartTime' in res.columns
    # Admission start should be 0.0
    assert res.iloc[0]['RelStartTime'] == 0.0
    # +2.5 hours
    assert res.iloc[1]['RelStartTime'] == 2.5
    # +24.0 hours
    assert res.iloc[2]['RelStartTime'] == 24.0
    # Check duration calc for 'A' (1 hour duration -> RelEndTime should be 3.5)
    assert res.iloc[1]['RelEndTime'] == 3.5

@pytest.mark.order(1)
def test_add_parent_raw_concepts(tmp_path, mock_tak_repo, base_ctx):
    """
    Test resolving parent concepts via the TAK hierarchy (MockRepo).
    """
    df = pd.DataFrame({
        'PatientID': [1, 1, 1, 1],
        # We simulate 'Concept' column existence as it is usually created by prior steps
        'Concept': ['A', 'B', 'C', 'Pattern'], 
        # Dummy cols required for DataProcessor init
        'ConceptName': ['A', 'B', 'C', 'Pattern'],
        'StartDateTime': [pd.Timestamp('2020-01-01')] * 4,
        'EndDateTime': [pd.Timestamp('2020-01-01')] * 4,
        'Value': [1, 1, 1, 1]
    })

    proc = DataProcessor(df, base_ctx, tak_repo_path=mock_tak_repo, checkpoint_path=str(tmp_path))
    proc.df = df # Override internal df 
    proc._add_parent_raw_concepts()
    
    res = proc.df
    
    # A -> A (Raw concept maps to self)
    assert res.iloc[0]['ParentRawConcepts'] == ['A']
    # B -> A (Single derivation)
    assert res.iloc[1]['ParentRawConcepts'] == ['A']
    # C -> B -> A (Nested derivation)
    assert res.iloc[2]['ParentRawConcepts'] == ['A']
    # Pattern -> A, D (Multiple parents, should be sorted)
    assert res.iloc[3]['ParentRawConcepts'] == ['A', 'D']

@pytest.mark.order(1)
def test_expand_tokens(tmp_path, mock_tak_repo, base_ctx):
    """
    Test splitting intervals into START/END tokens vs keeping instant events.
    """
    df = pd.DataFrame({
        'PatientID': [1, 1],
        'ConceptName': ['LongInterval', 'InstantEvent'],
        'StartDateTime': [
            pd.Timestamp('2020-01-01 10:00:00'),
            pd.Timestamp('2020-01-01 12:00:00')
        ],
        'EndDateTime': [
            pd.Timestamp('2020-01-01 11:00:00'), # 1 hour duration
            pd.Timestamp('2020-01-01 12:00:00')  # 0 duration
        ],
        'Value': ['High', 'True'],
        # Pre-calculated relative times required by _expand_tokens
        'RelStartTime': [0.0, 2.0],
        'RelEndTime': [1.0, 2.0]
    })

    proc = DataProcessor(df, base_ctx, tak_repo_path=mock_tak_repo, checkpoint_path=str(tmp_path))
    proc.df = df
    proc._expand_tokens(min_interval_duration_sec=1)
    res = proc.df

    # LongInterval should become 2 tokens (START, END)
    long_tokens = res[res['Concept'] == 'LongInterval']
    assert len(long_tokens) == 2
    assert long_tokens.iloc[0]['PositionToken'] == 'LongInterval_High_START'
    assert long_tokens.iloc[1]['PositionToken'] == 'LongInterval_High_END'
    
    # InstantEvent should be 1 token
    instant_tokens = res[res['Concept'] == 'InstantEvent']
    assert len(instant_tokens) == 1
    # 'True' value is usually omitted in base token name logic
    assert instant_tokens.iloc[0]['PositionToken'] == 'InstantEvent'

@pytest.mark.order(1)
def test_insert_null_tokens(tmp_path, mock_tak_repo, base_ctx):
    """
    Test insertion of [NULL] token in large gaps when no intervals are open.
    Fixed: Used two separate instant events to ensure open_stack==0 during the gap.
    """
    df = pd.DataFrame({
        'PatientID': [1, 1],
        # Scenario: Two events separated by 10 hours.
        # Event 1 at T=0.0
        # Event 2 at T=10.0
        # Gap = 10.0 > 3.0
        'PositionToken': ['Event_A', 'Event_B'],
        'TimePoint': [
            0.0, 
            10.0 
        ],
        # Pass-through dummy columns
        'RawConcept': ['A', 'B'],
        'Concept': ['A', 'B'],
        'ValueToken': ['A', 'B'],
        'StartDateTime': [pd.Timestamp('2020-01-01 10:00'), pd.Timestamp('2020-01-01 20:00')],
        'EndDateTime': [pd.Timestamp('2020-01-01 10:00'), pd.Timestamp('2020-01-01 20:00')],
        'Value': [1, 1],
        'ConceptName': ['A', 'B']
    })

    proc = DataProcessor(df, base_ctx, tak_repo_path=mock_tak_repo, checkpoint_path=str(tmp_path))
    proc.df = df
    # Insert NULLs for gaps > 3 hours
    proc._insert_null_tokens(gap_hrs=3)
    res = proc.df

    # Expect: Event_A, [NULL], Event_B
    assert len(res) == 3
    assert res.iloc[1]['Concept'] == '[NULL]'
    # Timepoint should be midpoint (5.0)
    assert res.iloc[1]['TimePoint'] == 5.0

@pytest.mark.order(1)
def test_cut_after_k_days(tmp_path, mock_tak_repo, base_ctx):
    """
    Test filtering logic:
    1. Drop visits completely if they are shorter than K days.
    2. For valid visits, cut all events happening after K days.
    """
    df = pd.DataFrame({
        'PatientID': [1, 1, 1, 2],
        'TimePoint': [
            24.0,       # Day 1 (Patient 1) - Kept
            25.0,       # Day 1 + 1hr (Patient 1) - Kept (Need >1 events to survive filtering)
            240.0,      # Day 10 (Patient 1) - Cut
            24.0        # Day 1 (Patient 2 - Short Visit) - Dropped entirely
        ],
        'Concept': ['A', 'B', 'C', 'A'],
        # Dummy cols
        'StartDateTime': [pd.Timestamp('2020-01-01')] * 4,
        'EndDateTime': [pd.Timestamp('2020-01-01')] * 4,
        'Value': [1, 1, 1, 1],
        'ConceptName': ['A', 'B', 'C', 'A']
    })

    # max_input_days = 5 days (120 hours)
    proc = DataProcessor(df, base_ctx, tak_repo_path=mock_tak_repo, max_input_days=5, checkpoint_path=str(tmp_path))
    proc.df = df
    proc._cut_after_k_days()
    res = proc.df

    # Patient 2 should be dropped (Max time 24h < 120h threshold)
    assert 2 not in res['PatientID'].values
    
    # Patient 1 should remain
    p1 = res[res['PatientID'] == 1]
    
    # The Day 10 event (240.0) should be cut
    # The Day 1 events (24.0 and 25.0) should remain
    assert len(p1) == 2
    assert p1.iloc[0]['TimePoint'] == 24.0
    assert p1.iloc[1]['TimePoint'] == 25.0