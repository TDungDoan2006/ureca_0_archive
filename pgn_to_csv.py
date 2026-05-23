#!/opt/anaconda3/bin/python

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import chess.pgn
except ModuleNotFoundError as exc:
    if exc.name == "chess":
        print("Missing dependency: python-chess")
        print(f"Current interpreter: {sys.executable}")
        print("Install it into this exact interpreter with:")
        print(f"  {sys.executable} -m pip install python-chess")
        raise SystemExit(1)
    raise


# Common PGN headers that are frequently useful in downstream analysis.
STANDARD_HEADER_FIELDS = [
    "Event",
    "Site",
    "Date",
    "Round",
    "White",
    "Black",
    "Result",
    "WhiteElo",
    "BlackElo",
    "WhiteTitle",
    "BlackTitle",
    "WhiteRatingDiff",
    "BlackRatingDiff",
    "WhiteFideId",
    "BlackFideId",
    "UTCDate",
    "UTCTime",
    "TimeControl",
    "Termination",
    "ECO",
    "Opening",
    "Variation",
    "SubVariation",
    "Annotator",
]


def parse_args():
    default_input = r"/Users/phamhoanghai/Ureca/dataset/lichess_db_standard_rated_2014-10.pgn"
    default_output = r"/Users/phamhoanghai/Ureca/dataset/output_data_v3.csv"

    parser = argparse.ArgumentParser(
        description="Extract PGN headers and move information into a CSV file."
    )
    parser.add_argument(
        "input_pgn",
        nargs="?",
        default=default_input,
        help="Path to the input PGN file.",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default=default_output,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of games to extract.",
    )
    return parser.parse_args()


def normalize_value(value):
    if value is None:
        return ""
    return str(value).strip()


def resolve_game_date(headers):
    date_value = normalize_value(headers.get("Date", ""))
    utc_date_value = normalize_value(headers.get("UTCDate", ""))

    if date_value and date_value != "????.??.??":
        return date_value, "Date"
    if utc_date_value and utc_date_value != "????.??.??":
        return utc_date_value, "UTCDate"
    return "", ""


def result_to_score(result):
    mapping = {
        "1-0": 1.0,
        "0-1": 0.0,
        "1/2-1/2": 0.5,
    }
    return mapping.get(result, "")


def safe_int(value):
    text = normalize_value(value)
    if text in {"", "?"}:
        return ""
    try:
        return int(text)
    except ValueError:
        return ""


def parse_time_control(value):
    text = normalize_value(value)
    if not text or text in {"?", "-", "unlimited"}:
        return {
            "TimeControlBaseSeconds": "",
            "TimeControlIncrementSeconds": "",
            "TimeControlBucket": "unknown",
        }

    if "+" in text:
        base_text, inc_text = text.split("+", 1)
    else:
        base_text, inc_text = text, "0"

    try:
        base_seconds = int(base_text)
        increment_seconds = int(inc_text)
    except ValueError:
        return {
            "TimeControlBaseSeconds": "",
            "TimeControlIncrementSeconds": "",
            "TimeControlBucket": "unknown",
        }

    estimated_total = base_seconds + 40 * increment_seconds
    if estimated_total < 180:
        bucket = "bullet"
    elif estimated_total < 600:
        bucket = "blitz"
    elif estimated_total < 1800:
        bucket = "rapid"
    else:
        bucket = "classical"

    return {
        "TimeControlBaseSeconds": base_seconds,
        "TimeControlIncrementSeconds": increment_seconds,
        "TimeControlBucket": bucket,
    }


def build_move_features(game):
    board = game.board()
    white_castled = False
    black_castled = False
    white_castle_side = "none"
    black_castle_side = "none"
    white_first_castle_ply = "none"
    black_first_castle_ply = "none"
    white_early_queen_move = 0
    black_early_queen_move = 0
    first_capture_ply = ""
    first_check_ply = ""
    opening_capture_count_12ply = 0
    opening_check_count_12ply = 0
    opening_piece_move_count_12ply = 0
    opening_pawn_move_count_12ply = 0
    white_pawn_moves = 0
    black_pawn_moves = 0
    white_knight_moves = 0
    black_knight_moves = 0
    white_bishop_moves = 0
    black_bishop_moves = 0
    white_rook_moves = 0
    black_rook_moves = 0
    white_queen_moves = 0
    black_queen_moves = 0
    white_king_moves = 0
    black_king_moves = 0
    total_captures = 0
    total_checks = 0
    total_promotions = 0
    en_passant_count = 0

    for ply_index, move in enumerate(game.mainline_moves(), start=1):
        mover_is_white = board.turn == chess.WHITE
        moving_piece = board.piece_at(move.from_square)
        is_capture = board.is_capture(move)
        is_en_passant = board.is_en_passant(move)
        is_castling = board.is_castling(move)
        is_promotion = move.promotion is not None
        san = board.san(move)
        gives_check = "+" in san or "#" in san

        if moving_piece is not None:
            if moving_piece.piece_type == chess.PAWN:
                if mover_is_white:
                    white_pawn_moves += 1
                else:
                    black_pawn_moves += 1
            elif moving_piece.piece_type == chess.KNIGHT:
                if mover_is_white:
                    white_knight_moves += 1
                else:
                    black_knight_moves += 1
            elif moving_piece.piece_type == chess.BISHOP:
                if mover_is_white:
                    white_bishop_moves += 1
                else:
                    black_bishop_moves += 1
            elif moving_piece.piece_type == chess.ROOK:
                if mover_is_white:
                    white_rook_moves += 1
                else:
                    black_rook_moves += 1
            elif moving_piece.piece_type == chess.QUEEN:
                if mover_is_white:
                    white_queen_moves += 1
                else:
                    black_queen_moves += 1
            elif moving_piece.piece_type == chess.KING:
                if mover_is_white:
                    white_king_moves += 1
                else:
                    black_king_moves += 1

            if ply_index <= 12:
                if moving_piece.piece_type == chess.PAWN:
                    opening_pawn_move_count_12ply += 1
                else:
                    opening_piece_move_count_12ply += 1

            if (
                moving_piece.piece_type == chess.QUEEN
                and ply_index <= 10
            ):
                if mover_is_white:
                    white_early_queen_move = 1
                else:
                    black_early_queen_move = 1

        if is_capture:
            total_captures += 1
            if ply_index <= 12:
                opening_capture_count_12ply += 1
            if first_capture_ply == "":
                first_capture_ply = ply_index

        if gives_check:
            total_checks += 1
            if ply_index <= 12:
                opening_check_count_12ply += 1
            if first_check_ply == "":
                first_check_ply = ply_index

        if is_en_passant:
            en_passant_count += 1

        if is_promotion:
            total_promotions += 1

        if is_castling:
            if chess.square_file(move.to_square) == 6:
                castle_side = "kingside"
            else:
                castle_side = "queenside"

            if mover_is_white and not white_castled:
                white_castled = True
                white_castle_side = castle_side
                white_first_castle_ply = ply_index
            elif not mover_is_white and not black_castled:
                black_castled = True
                black_castle_side = castle_side
                black_first_castle_ply = ply_index

        board.push(move)

    ply_count = len(board.move_stack)
    move_count = (ply_count + 1) // 2
    game_length_bucket = (
        "short" if ply_count < 40 else
        "medium" if ply_count < 80 else
        "long"
    )

    return {
        "PlyCount": ply_count,
        "MoveCount": move_count,
        "GameLengthBucket": game_length_bucket,
        "WhiteCastled": int(white_castled),
        "BlackCastled": int(black_castled),
        "WhiteCastleSide": white_castle_side,
        "BlackCastleSide": black_castle_side,
        "WhiteFirstCastlePly": white_first_castle_ply,
        "BlackFirstCastlePly": black_first_castle_ply,
        "WhiteEarlyQueenMove": white_early_queen_move,
        "BlackEarlyQueenMove": black_early_queen_move,
        "FirstCapturePly": first_capture_ply,
        "FirstCheckPly": first_check_ply,
        "OpeningCaptureCount12Ply": opening_capture_count_12ply,
        "OpeningCheckCount12Ply": opening_check_count_12ply,
        "OpeningPieceMoveCount12Ply": opening_piece_move_count_12ply,
        "OpeningPawnMoveCount12Ply": opening_pawn_move_count_12ply,
        "WhitePawnMoves": white_pawn_moves,
        "BlackPawnMoves": black_pawn_moves,
        "WhiteKnightMoves": white_knight_moves,
        "BlackKnightMoves": black_knight_moves,
        "WhiteBishopMoves": white_bishop_moves,
        "BlackBishopMoves": black_bishop_moves,
        "WhiteRookMoves": white_rook_moves,
        "BlackRookMoves": black_rook_moves,
        "WhiteQueenMoves": white_queen_moves,
        "BlackQueenMoves": black_queen_moves,
        "WhiteKingMoves": white_king_moves,
        "BlackKingMoves": black_king_moves,
        "TotalCaptures": total_captures,
        "TotalChecks": total_checks,
        "TotalPromotions": total_promotions,
        "EnPassantCount": en_passant_count,
    }


def get_move_lists(game):
    board = game.board()
    san_moves = []
    uci_moves = []

    for move in game.mainline_moves():
        san_moves.append(board.san(move))
        uci_moves.append(move.uci())
        board.push(move)

    return san_moves, uci_moves


def build_row(game, game_index):
    headers = {key: normalize_value(value) for key, value in game.headers.items()}
    san_moves, uci_moves = get_move_lists(game)
    resolved_date, resolved_date_source = resolve_game_date(headers)
    move_features = build_move_features(game)
    time_control_features = parse_time_control(headers.get("TimeControl", ""))

    white_elo_num = safe_int(headers.get("WhiteElo", ""))
    black_elo_num = safe_int(headers.get("BlackElo", ""))
    white_rating_diff_num = safe_int(headers.get("WhiteRatingDiff", ""))
    black_rating_diff_num = safe_int(headers.get("BlackRatingDiff", ""))

    extra_headers = {
        key: value for key, value in headers.items() if key not in STANDARD_HEADER_FIELDS
    }

    row = {
        "GameID": game_index,
        "MovesSAN": " ".join(san_moves),
        "MovesUCI": " ".join(uci_moves),
        "FinalFEN": game.end().board().fen(),
        "EloDiff": (
            white_elo_num - black_elo_num
            if isinstance(white_elo_num, int) and isinstance(black_elo_num, int)
            else ""
        ),
        "AvgElo": (
            (white_elo_num + black_elo_num) / 2
            if isinstance(white_elo_num, int) and isinstance(black_elo_num, int)
            else ""
        ),
        "GameDate": resolved_date,
        "GameDateSource": resolved_date_source,
        "OpeningCode": headers.get("ECO", ""),
        "OpeningName": headers.get("Opening", ""),
        "NumericResult": result_to_score(headers.get("Result", "")),
        "ExtraHeaders": json.dumps(extra_headers, ensure_ascii=True, sort_keys=True),
    }
    row.update(time_control_features)
    row.update(move_features)

    for field in STANDARD_HEADER_FIELDS:
        row[field] = headers.get(field, "")

    row["WhiteElo"] = white_elo_num
    row["BlackElo"] = black_elo_num
    row["WhiteRatingDiff"] = white_rating_diff_num
    row["BlackRatingDiff"] = black_rating_diff_num

    if not row["Date"] or row["Date"] == "????.??.??":
        row["Date"] = resolved_date

    return row


def extract_pgn_to_csv(input_pgn, output_csv, limit=None):
    input_path = Path(input_pgn)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        ["GameID"]
        + STANDARD_HEADER_FIELDS
        + [
            "EloDiff",
            "AvgElo",
            "GameDate",
            "GameDateSource",
            "OpeningCode",
            "OpeningName",
            "NumericResult",
            "TimeControlBaseSeconds",
            "TimeControlIncrementSeconds",
            "TimeControlBucket",
            "PlyCount",
            "MoveCount",
            "GameLengthBucket",
            "WhiteCastled",
            "BlackCastled",
            "WhiteCastleSide",
            "BlackCastleSide",
            "WhiteFirstCastlePly",
            "BlackFirstCastlePly",
            "WhiteEarlyQueenMove",
            "BlackEarlyQueenMove",
            "FirstCapturePly",
            "FirstCheckPly",
            "OpeningCaptureCount12Ply",
            "OpeningCheckCount12Ply",
            "OpeningPieceMoveCount12Ply",
            "OpeningPawnMoveCount12Ply",
            "WhitePawnMoves",
            "BlackPawnMoves",
            "WhiteKnightMoves",
            "BlackKnightMoves",
            "WhiteBishopMoves",
            "BlackBishopMoves",
            "WhiteRookMoves",
            "BlackRookMoves",
            "WhiteQueenMoves",
            "BlackQueenMoves",
            "WhiteKingMoves",
            "BlackKingMoves",
            "TotalCaptures",
            "TotalChecks",
            "TotalPromotions",
            "EnPassantCount",
            "MovesSAN",
            "MovesUCI",
            "FinalFEN",
            "ExtraHeaders",
        ]
    )

    extracted = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as pgn_file, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        game_index = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            game_index += 1
            writer.writerow(build_row(game, game_index))
            extracted += 1

            if extracted % 10000 == 0:
                print(f"Extracted {extracted:,} games...")

            if limit is not None and extracted >= limit:
                break

    print(f"Finished. Extracted {extracted:,} games to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    extract_pgn_to_csv(args.input_pgn, args.output_csv, limit=args.limit)
