from dataclasses import dataclass


# Simple phase split for 8x8 Gomoku (max 64 moves)
# Adjust later if you want finer bands
def phase_from_move(move_number: int) -> str:
    if move_number < 12:  # opening
        return "early"
    elif move_number < 36:  # midgame
        return "mid"
    else:  # endgame
        return "late"


def outcome_from_z(z: float) -> str:
    if z > 0:
        return "win"
    if z < 0:
        return "loss"
    return "draw"


@dataclass(frozen=True)
class BucketKey:
    outcome: str  # win/loss/draw
    phase: str  # early/mid/late


def bucket_key(move_number: int, z: float) -> BucketKey:
    return BucketKey(outcome=outcome_from_z(z), phase=phase_from_move(move_number))
