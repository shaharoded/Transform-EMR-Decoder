import pandas as pd
from torch.utils.data import DataLoader
import pytest

from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset, collate_emr


@pytest.mark.order(1)
def test_synthetic_data_pipeline(tmp_path, capsys):
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
    proc_t = DataProcessor(df, ctx, max_input_days=1)
    temporal_df_t, _ = proc_t.run()
    proc = DataProcessor(df, ctx)
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
    ds = EMRDataset(temporal_df, context_df, tokenizer=tokenizer)

    # collate one batch of size 2
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_emr)
    batch = next(iter(dl))

    # expecting these tensors in batch and correct batch dimension
    expected_keys = (
        'raw_concept_ids', 'concept_ids', 'value_ids',
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