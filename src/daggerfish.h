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

#ifndef DAGGERFISH_H_INCLUDED
#define DAGGERFISH_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <string>
#include <tuple>
#include <vector>
#include <memory>

#include "types.h"

namespace Stockfish {

class Position;
class ThreadPool;

namespace Daggerfish {

constexpr int GraphSlotCount = 4;
constexpr int RepContextKeyCount = 128;

struct GraphEntry;
class GraphTable;
struct RepPseudoEntry;
struct InflightCell;

enum class PublicationGate : uint8_t {
    VerifiedGlobal,
    RepetitionSensitive,
    LocalOnly
};

enum class RepetitionKind : uint8_t {
    None,
    HasRepeated,
    Upcoming,
    HighRule50
};

enum class GraphNodeState : uint8_t {
    Idle,
    Solving,
    Solved,
    RepSensitive,
    Abandoned
};

enum class HintKind : uint8_t {
    NullMove,
    ProbCut,
    Razoring,
    Futility,
    LMR,
    Singular,
    QSearchStandPat,
    Other
};

enum class PublishSource : uint8_t {
    MainSearch,
    QSearch,
    Tablebase
};

struct RepContext {
    Key      keys[RepContextKeyCount] = {};
    Key      pathSig                  = 0;
    uint16_t count                    = 0;
    uint16_t overflow                 = 0;
    uint16_t rule50                   = 0;
};

struct GraphData {
    Move  move;
    Value lower, upper, eval;
    Depth depth;
    bool  is_pv;

    GraphData() = delete;

    GraphData(Move m, Value l, Value u, Value ev, Depth d, bool pv) :
        move(m),
        lower(l),
        upper(u),
        eval(ev),
        depth(d),
        is_pv(pv) {}
};

struct GraphSlot {
    int16_t lower16 = VALUE_NONE;
    int16_t upper16 = VALUE_NONE;
    int16_t eval16  = VALUE_NONE;
    Move    move16  = Move::none();
    uint8_t depth8  = 0;
    uint8_t genPv8  = 0;

    bool      occupied() const;
    uint8_t   relative_age(uint8_t generation8) const;
    GraphData read() const;
    void      save(Value lower, Value upper, bool pv, Depth d, Move m, Value ev, uint8_t generation8);
};

struct GraphEntry {
    Key       key  = 0;
    uint8_t   used = 0;
    GraphSlot slots[GraphSlotCount];

    void reset(Key k);
};

struct RepPseudoEntry {
    Key            key      = 0;
    Key            baseKey  = 0;
    Key            pathSig  = 0;
    uint16_t       rule50   = 0;
    uint8_t        used     = 0;
    RepetitionKind kind     = RepetitionKind::None;
    GraphSlot      slot;

    void reset(Key k, Key base, const RepContext& ctx, RepetitionKind repKind);
};

struct GraphStats {
    uint64_t canonicalProbes       = 0;
    uint64_t canonicalHits         = 0;
    uint64_t canonicalStores       = 0;
    uint64_t graphCutoffs          = 0;
    uint64_t blockedRepetition     = 0;
    uint64_t blockedHighRule50     = 0;
    uint64_t pseudoProbes          = 0;
    uint64_t pseudoHits            = 0;
    uint64_t pseudoStores          = 0;
    uint64_t incompatiblePathSkips = 0;
    uint64_t inflightClaims        = 0;
    uint64_t inflightReleases      = 0;
    uint64_t inflightWaits         = 0;
    uint64_t inflightWaitHits      = 0;
    uint64_t inflightFallbacks     = 0;
    uint64_t inflightWakeups       = 0;
    uint64_t inflightAbandoned     = 0;
    uint64_t verifiedStores        = 0;
    uint64_t localHints            = 0;
    uint64_t rejectedSpeculative   = 0;
};

struct InflightCell {
    std::atomic<Key>      key {0};
    std::atomic<int16_t>  depth {0};
    std::atomic<uint16_t> owner {0};
    std::atomic<uint32_t> waiters {0};
    std::atomic<uint32_t> sequence {0};
    std::atomic<uint8_t>  state {uint8_t(GraphNodeState::Idle)};

    void clear();
};

class InflightGuard {
   public:
    InflightGuard();
    InflightGuard(const InflightGuard&)            = delete;
    InflightGuard& operator=(const InflightGuard&) = delete;
    InflightGuard(InflightGuard&& other) noexcept;
    InflightGuard& operator=(InflightGuard&& other) noexcept;
    ~InflightGuard();

    bool active() const;
    void release();

   private:
    friend class GraphTable;
    InflightGuard(GraphTable* graph, size_t slot, Key key, Depth depth, uint16_t owner);

    GraphTable* graphTable = nullptr;
    size_t      slot       = 0;
    Key         key        = 0;
    int16_t     depth      = 0;
    uint16_t    owner      = 0;
};

class GraphTable {
   public:
    void resize(size_t mbSize, ThreadPool& threads);
    void clear(ThreadPool& threads);

    void    new_search();
    uint8_t generation() const;
    bool    enabled() const;

    std::tuple<bool, GraphData>              probe(Key key, Depth depth) const;
    std::tuple<bool, GraphData>              probe_repetition(Key              baseKey,
                                                              Depth            depth,
                                                              const RepContext& ctx,
                                                              RepetitionKind    kind) const;
    void                                     store_verified(Key           key,
                                                            Value         v,
                                                            bool          pv,
                                                            Bound         b,
                                                            Depth         d,
                                                            Move          m,
                                                            Value         ev,
                                                            uint8_t       generation8,
                                                            PublishSource source);
    void                                     store_repetition(Key              baseKey,
                                                             Value            v,
                                                             bool             pv,
                                                             Bound            b,
                                                             Depth            d,
                                                             Move             m,
                                                             Value            ev,
                                                             const RepContext& ctx,
                                                             RepetitionKind    kind);
    void                                     store_local_hint(Key key,
                                                             Value v,
                                                             Bound b,
                                                             Depth d,
                                                             Move m,
                                                             HintKind kind);
    InflightGuard                           begin_inflight(Key key, Depth depth, uint16_t owner);
    std::tuple<bool, GraphData>             wait_for_inflight(Key      key,
                                                              Depth    depth,
                                                              uint16_t owner,
                                                              Value    alpha,
                                                              Value    beta,
                                                              int      spins) const;
    void                                     record_cutoff();
    void                                     record_blocked(RepetitionKind kind);
    GraphStats                               stats() const;
    std::string                              stats_string() const;
    int                                      hashfull(int maxAge = 0) const;

   private:
    friend class InflightGuard;

    GraphEntry* first_entry(Key key) const;
    RepPseudoEntry* first_rep_entry(Key key) const;
    InflightCell* first_inflight(Key key) const;
    void save_slot(GraphEntry& entry,
                   Key         key,
                   Value       v,
                   bool        pv,
                   Bound       b,
                   Depth       d,
                   Move        m,
                   Value       ev,
                   uint8_t     generation8);
    void complete_inflight(Key key, Depth depth, GraphNodeState state);
    void release_inflight(size_t slot, Key key, int16_t depth, uint16_t owner);
    void reset_stats();

    std::vector<GraphEntry> entries;
    std::vector<RepPseudoEntry> repEntries;
    std::unique_ptr<InflightCell[]> inflightEntries;
    size_t inflightEntryCount = 0;
    uint8_t                 generation8 = 0;

    mutable std::atomic<uint64_t> canonicalProbes {0};
    mutable std::atomic<uint64_t> canonicalHits {0};
    mutable std::atomic<uint64_t> canonicalStores {0};
    mutable std::atomic<uint64_t> graphCutoffs {0};
    mutable std::atomic<uint64_t> blockedRepetition {0};
    mutable std::atomic<uint64_t> blockedHighRule50 {0};
    mutable std::atomic<uint64_t> pseudoProbes {0};
    mutable std::atomic<uint64_t> pseudoHits {0};
    mutable std::atomic<uint64_t> pseudoStores {0};
    mutable std::atomic<uint64_t> incompatiblePathSkips {0};
    mutable std::atomic<uint64_t> inflightClaims {0};
    mutable std::atomic<uint64_t> inflightReleases {0};
    mutable std::atomic<uint64_t> inflightWaits {0};
    mutable std::atomic<uint64_t> inflightWaitHits {0};
    mutable std::atomic<uint64_t> inflightFallbacks {0};
    mutable std::atomic<uint64_t> inflightWakeups {0};
    mutable std::atomic<uint64_t> inflightAbandoned {0};
    mutable std::atomic<uint64_t> verifiedStores {0};
    mutable std::atomic<uint64_t> localHints {0};
    mutable std::atomic<uint64_t> rejectedSpeculative {0};
};

RepContext make_rep_context(const Position& pos);
Key        repetition_key(Key baseKey, const RepContext& ctx, RepetitionKind kind);

}  // namespace Daggerfish
}  // namespace Stockfish

#endif  // #ifndef DAGGERFISH_H_INCLUDED
