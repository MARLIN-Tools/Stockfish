/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "pawnhash.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "bitboard.h"
#include "memory.h"
#include "position.h"

namespace Stockfish {

namespace PawnHash {

namespace {

struct Entry {
    Key          key;
    std::int16_t score;
    std::int16_t padding;
};

static constexpr int ClusterSize = 4;

struct Cluster {
    Entry entry[ClusterSize];
};

static_assert(sizeof(Cluster) == 64, "Suboptimal pawn hash cluster size");

Cluster*     table        = nullptr;
std::size_t  clusterCount = 0;

constexpr int PassedBonus[8] = {0, 4, 8, 14, 24, 40, 0, 0};
constexpr int DoubledPenalty  = 6;
constexpr int IsolatedPenalty = 8;

Bitboard adjacent_files_bb(File f) {
    Bitboard result = 0;

    if (f > FILE_A)
        result |= file_bb(File(f - 1));
    if (f < FILE_H)
        result |= file_bb(File(f + 1));

    return result;
}

Bitboard passed_pawn_mask(Color c, Square s) {
    Bitboard files = file_bb(s) | adjacent_files_bb(file_of(s));
    Rank     r     = rank_of(s);

    if (c == WHITE)
    {
        if (r == RANK_8)
            return 0;

        return files & (~Bitboard(0) << (8 * (r + 1)));
    }

    if (r == RANK_1)
        return 0;

    return files & ((Bitboard(1) << (8 * r)) - 1);
}

int evaluate_color(const Position& pos, Color c) {
    int      score = 0;
    Bitboard pawns = pos.pieces(c, PAWN);
    Bitboard enemyPawns = pos.pieces(~c, PAWN);

    for (File f = FILE_A; f <= FILE_H; ++f)
    {
        int fileCount = popcount(pawns & file_bb(f));
        if (fileCount > 1)
            score -= DoubledPenalty * (fileCount - 1);
    }

    while (pawns)
    {
        Square s = pop_lsb(pawns);
        File   f = file_of(s);

        if (!(pos.pieces(c, PAWN) & adjacent_files_bb(f)))
            score -= IsolatedPenalty;

        if (!(enemyPawns & passed_pawn_mask(c, s)))
            score += PassedBonus[relative_rank(c, s)];
    }

    return score;
}

int evaluate_pawn_structure(const Position& pos) {
    return evaluate_color(pos, WHITE) - evaluate_color(pos, BLACK);
}

Entry* first_entry(const Key key) {
    return &table[mul_hi64(key, clusterCount)].entry[0];
}

}  // namespace

void resize(std::size_t mbSize) {
    aligned_large_pages_free(table);
    table = nullptr;
    clusterCount = 0;

    if (mbSize == 0)
        return;

    clusterCount = std::max<std::size_t>(1, mbSize * 1024 * 1024 / sizeof(Cluster));
    table        = static_cast<Cluster*>(aligned_large_pages_alloc(clusterCount * sizeof(Cluster)));

    if (!table)
    {
        std::cerr << "Failed to allocate " << mbSize << "MB for pawn hash." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    clear();
}

void clear() {
    if (table)
        std::memset(table, 0, clusterCount * sizeof(Cluster));
}

int probe(const Position& pos) {
    if (!table)
        return 0;

    Entry* const cluster = first_entry(pos.pawn_key());

    for (int i = 0; i < ClusterSize; ++i)
        if (cluster[i].key == pos.pawn_key())
            return cluster[i].score;

    int score = evaluate_pawn_structure(pos);
    Entry* replace = &cluster[0];

    for (int i = 0; i < ClusterSize; ++i)
        if (cluster[i].key == 0)
        {
            replace = &cluster[i];
            break;
        }

    replace->key   = pos.pawn_key();
    replace->score = static_cast<std::int16_t>(score);

    return score;
}

}  // namespace PawnHash

}  // namespace Stockfish
