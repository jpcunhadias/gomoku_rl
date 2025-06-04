from game.gomoku import GomokuGameManager
from game.player import HumanPlayer, RandomPlayer


def play_human_vs_random() -> None:
    """Run a full game between a human and a random agent."""

    game = GomokuGameManager()
    human = HumanPlayer(1)
    ai = RandomPlayer(2)

    game.reset()

    while not game.is_over():
        game.render()
        player = human if game.board.current_player == 1 else ai
        move = player.get_action(game.board)
        game.play_move(*move)

    game.render()
    winner = game.get_winner()
    print("Winner:", "Draw" if winner is None else f"Player {winner}")


if __name__ == "__main__":
    play_human_vs_random()
