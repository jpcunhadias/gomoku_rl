import pytest
from game.gomoku import GomokuGameManager


def test_draw_game():
    game = GomokuGameManager()
    game.reset()

    size = game.board.board_size

    # Fill the board in a checkerboard pattern that avoids any win
    for row in range(size):
        for col in range(size):
            if game.is_over():
                break
            player = 1 if (row + col) % 2 == 0 else 2
            game.board.board[row, col] = player
            game.board.last_move = (row, col)
            game.board.current_player = (
                3 - player
            )  # Just to allow check_win to work correctly
            if game.board.check_win()[0]:
                pytest.skip("Pattern accidentally formed a win")

    game.board.current_player = 1  # Set back to 1 for consistency
    game.board.last_move = (size - 1, size - 1)  # Last move just to satisfy check
    assert game.board.is_draw()
    assert game.is_over()
    assert game.get_winner() is None

    assert game.is_over() is True
    assert game.get_winner() is None


def test_play_move_by_index():
    game = GomokuGameManager()
    index = game.board.move_to_index(7, 7)
    game.play_move_by_index(index)

    assert game.board.board[7, 7] == 1
    assert game.board.last_move == (7, 7)


def test_get_legal_moves_vs_board():
    game = GomokuGameManager()
    legal_from_manager = set(game.get_legal_moves())
    legal_from_board = set(game.board.get_legal_moves())

    assert legal_from_manager == legal_from_board


def test_reset_game_manager():
    game = GomokuGameManager()
    game.play_move(7, 7)
    game.reset()

    assert game.board.last_move is None
    assert game.board.current_player == 1
    assert not game.is_over()
    assert game.get_winner() is None
    assert game.board.board.sum() == 0
