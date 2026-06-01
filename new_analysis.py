import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score
import mne

#dropping noisy channels before epoch
DROP_CH = ['C3', 'TP9', 'Fp1', 'Fp2', 'Oz', 'O2']
raw.drop_channels([ch for ch in DROP_CH if ch in raw.ch_names])


events_orig, _ = mne.events_from_annotations(raw, verbose=False)

#epoch all events
epochs_speed = mne.Epochs(
    raw, events_orig,
    event_id={'Liv/speed': 12, 'Nonliv/speed': 13},
    tmin=-0.2, tmax=0.8,
    baseline=(None, 0),
    reject={'eeg': 150e-6},
    picks='eeg',
    preload=True
)

epochs_norm = mne.Epochs(
    raw, events_orig,
    event_id={'Liv/norm': 2, 'Nonliv/norm': 3},
    tmin=-0.2, tmax=0.8,
    baseline=(None, 0),
    reject={'eeg': 150e-6},
    picks='eeg',
    preload=True
)

print(f"Speed epochs after rejection: {len(epochs_speed)}")
print(f"  of which living:     {len(epochs_speed['Liv/speed'])}")
print(f"  of which non-living: {len(epochs_speed['Nonliv/speed'])}")
print(f"Norm epochs after rejection:  {len(epochs_norm)}")
print(f"  of which living:     {len(epochs_norm['Liv/norm'])}")
print(f"  of which non-living: {len(epochs_norm['Nonliv/norm'])}")

#block groups
speed_block_onsets = events_orig[events_orig[:, 2] == 8][:, 0]
norm_block_onsets  = events_orig[events_orig[:, 2] == 7][:, 0]

def assign_groups_fixed(epochs, block_onsets):
    sample_onsets = epochs.events[:, 0]
    groups = []
    for s in sample_onsets:
        prior = block_onsets[block_onsets <= s]
        block_idx = len(prior) - 1 if len(prior) > 0 else 0
        groups.append(block_idx)
    return np.array(groups)

groups_speed = assign_groups_fixed(epochs_speed, speed_block_onsets)
groups_norm  = assign_groups_fixed(epochs_norm,  norm_block_onsets)

print(f"\nUnique blocks in speed: {np.unique(groups_speed)}")
print(f"Unique blocks in norm:  {np.unique(groups_norm)}")
print(f"Speed group counts:", {g: (groups_speed==g).sum() for g in np.unique(groups_speed)})
print(f"Norm  group counts:", {g: (groups_norm==g).sum() for g in np.unique(groups_norm)})

#feature matrix
def bin_times(X, n_bins=20):
    n_epochs, n_ch, n_times = X.shape
    bin_size = n_times // n_bins
    trimmed = X[:, :, :n_bins * bin_size]
    return trimmed.reshape(n_epochs, n_ch, n_bins, bin_size).mean(axis=3)

X_speed = bin_times(epochs_speed.get_data()).reshape(len(epochs_speed), -1)
X_norm  = bin_times(epochs_norm.get_data()).reshape(len(epochs_norm),  -1)

X      = np.vstack([X_speed, X_norm])
y      = np.array([1] * len(epochs_speed) + [0] * len(epochs_norm))
groups = np.concatenate([groups_speed, groups_norm])

print(f"\nFeature matrix shape: {X.shape}")
print(f"Class balance — speed: {y.sum()}, norm: {(y==0).sum()}")
print(f"Number of unique groups (blocks): {len(np.unique(groups))}")

#logo CV
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42, C=0.1))
])

logo = LeaveOneGroupOut()
scores_clean = []
skipped = 0

for train_idx, test_idx in logo.split(X, y, groups):
    y_test = y[test_idx]
    if len(np.unique(y_test)) < 2:
        skipped += 1
        continue
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])
    scores_clean.append(balanced_accuracy_score(y_test, y_pred))

scores_clean = np.array(scores_clean)

print(f"Total folds:{len(scores_clean) + skipped}")
print(f"Folds skipped:{skipped} (test set had only one class)")
print(f"Folds used:{len(scores_clean)}")
print(f"Mean balanced accuracy: {scores_clean.mean():.4f}")
print(f"Std:{scores_clean.std():.4f}")
print(f"SE:{scores_clean.std() / np.sqrt(len(scores_clean)):.4f}")
print(f"Scores per fold:{np.round(scores_clean, 3)}")