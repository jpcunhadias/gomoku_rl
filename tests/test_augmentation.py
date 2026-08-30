import torch

from train.augmentation import TRANSFORMS, apply_random_transform, augment_data


def _marker_state_and_policy(row: int, col: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A state with a single stone and a policy with all mass on the same cell, so any
    transform's effect on each can be located and compared directly."""
    state = torch.zeros(3, 8, 8)
    state[0, row, col] = 1.0
    policy = torch.zeros(8, 8)
    policy[row, col] = 1.0
    return state, policy


def _marker_position(tensor: torch.Tensor) -> tuple[int, int]:
    nonzero = torch.nonzero(tensor)
    assert nonzero.shape[0] == 1, f"expected exactly one nonzero cell, got {nonzero.shape[0]}"
    row, col = nonzero[0].tolist()
    return row, col


def test_all_transforms_move_state_and_policy_markers_together():
    """The actual invariant that matters: whatever a transform does to the board, it must
    do the identical thing to the policy target, or training silently learns mislabeled
    (state, policy) pairs for every augmented sample using that transform."""
    # Off-diagonal, off-center so every one of the 6 transforms gives a genuinely different
    # position - a marker on a symmetry axis wouldn't distinguish a buggy transform from a
    # correct one.
    row, col = 1, 5

    for state_tf, policy_tf in TRANSFORMS:
        state, policy = _marker_state_and_policy(row, col)
        transformed_state = state_tf(state)
        transformed_policy = policy_tf(policy)

        assert transformed_state.shape == (3, 8, 8)
        assert transformed_policy.shape == (8, 8)

        state_pos = _marker_position(transformed_state[0])
        policy_pos = _marker_position(transformed_policy)
        assert state_pos == policy_pos, (
            f"state marker moved to {state_pos} but policy marker moved to {policy_pos} "
            f"for transform pair (state={state_tf}, policy={policy_tf})"
        )


def test_transforms_are_all_distinct_and_cover_the_dihedral_group():
    """6 transforms should give 6 different results for an asymmetric marker - if two
    transforms silently computed the same thing, half the intended augmentation variety
    would be missing without any test catching it."""
    row, col = 1, 5
    state, _ = _marker_state_and_policy(row, col)

    positions = set()
    for state_tf, _ in TRANSFORMS:
        positions.add(_marker_position(state_tf(state)[0]))

    assert len(positions) == len(TRANSFORMS)


def test_apply_random_transform_preserves_shapes_and_alignment():
    row, col = 2, 6
    state, policy = _marker_state_and_policy(row, col)

    for _ in range(50):
        aug_state, aug_policy = apply_random_transform(state.clone(), policy.clone())
        assert aug_state.shape == (3, 8, 8)
        assert aug_policy.shape == (8, 8)
        assert _marker_position(aug_state[0]) == _marker_position(aug_policy)


def test_augment_data_leaves_value_untouched():
    row, col = 3, 4
    state, policy = _marker_state_and_policy(row, col)
    data = [(state, policy, -1.0), (state.clone(), policy.clone(), 1.0)]

    augmented = augment_data(data)

    assert [z for _, _, z in augmented] == [-1.0, 1.0]
    for (aug_state, aug_policy, _), (_, _, _) in zip(augmented, data, strict=True):
        assert _marker_position(aug_state[0]) == _marker_position(aug_policy)
