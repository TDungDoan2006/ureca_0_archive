# Elo Coaching Feature Importance

This report trains a coaching-focused Elo model using player-controlled behaviors rather than time-control metadata.

## Validation

- Validation RMSE: 230.54
- Validation MAE: 182.57
- Validation R^2: 0.087
- Test RMSE: 234.97
- Test MAE: 185.34
- Test R^2: 0.089

## Top Actionable Signals

- `pawn_moves_rate`: importance `22852.02`, coefficient `109.42`. Higher is better. Higher values of this feature are associated with stronger players in this model.
- `king_moves_rate`: importance `19343.83`, coefficient `96.33`. Higher is better. Frequent king moves usually mean king safety problems. Castle earlier and avoid creating unnecessary king walks.
- `heavy_piece_move_rate`: importance `12051.50`, coefficient `76.66`. Higher is better. Frequent queen and rook moves can be good later, but too much heavy-piece activity early may signal wasted tempi.
- `minor_piece_move_rate`: importance `9410.03`, coefficient `69.52`. Higher is better. Using knights and bishops actively is generally healthier than overusing the queen early.
- `knight_moves_rate`: importance `5101.48`, coefficient `50.77`. Higher is better. Extra knight moves can mean rerouting, but too many early knight moves may also lose time. Aim for purposeful development rather than repetition.
- `queen_moves_rate`: importance `4364.78`, coefficient `46.69`. Higher is better. A high share of queen moves often means lost tempi or repeated queen chases. Try to solve positions with minor pieces first.
- `rook_moves_rate`: importance `4293.93`, coefficient `46.57`. Higher is better. More rook activity can reflect better development and open-file usage. Bring rooks into play after king safety and minor-piece development.
- `bishop_moves_rate`: importance `3619.76`, coefficient `42.88`. Higher is better. Active bishops often support stronger positions. Look for simple developing squares before starting direct attacks.
- `checks_per_move`: importance `1293.71`, coefficient `-24.62`. Lower is better. A lower check rate is fine if your moves improve development, king safety, and material balance.
- `MoveCount`: importance `1162.35`, coefficient `24.62`. Higher is better. Higher values of this feature are associated with stronger players in this model.

## Simple Beginner Examples

- `early_queen_move`: average Elo changes from `1645.0` to `1569.1` (gap `-76.0`) between values `0` and `1`.
- `castled`: average Elo changes from `1555.7` to `1657.4` (gap `101.7`) between values `0` and `1`.
