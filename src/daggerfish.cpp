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

#include "daggerfish.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>
#include <thread>

#include "position.h"
#include "thread.h"

namespace Stockfish::Daggerfish {

namespace {

constexpr unsigned GENERATION_BITS  = 3;
constexpr int      GENERATION_DELTA = (1 << GENERATION_BITS);
constexpr int      GENERATION_CYCLE = 255 + GENERATION_DELTA;
constexpr int      GENERATION_MASK  = (0xFF << GENERATION_BITS) & 0xFF;

constexpr Value GRAPH_LOWER_UNKNOWN = -VALUE_INFINITE;
constexpr Value GRAPH_UPPER_UNKNOWN = VALUE_INFINITE;
constexpr Key   REP_HASH_SEED       = 0x9E3779B97F4A7C15ULL;

Value clamp_to_i16(Value v) {
    return std::clamp(v, int(INT16_MIN), int(INT16_MAX));
}

std::pair<Value, Value> interval_from_bound(Value v, Bound b) {
    switch (b)
    {
    case BOUND_EXACT :
        return {v, v};
    case BOUND_LOWER :
        return {v, GRAPH_UPPER_UNKNOWN};
    case BOUND_UPPER :
        return {GRAPH_LOWER_UNKNOWN, v};
    default :
        return {GRAPH_LOWER_UNKNOWN, GRAPH_UPPER_UNKNOWN};
    }
}

Key mix_key(Key k) {
    k ^= k >> 30;
    k *= 0xBF58476D1CE4E5B9ULL;
    k ^= k >> 27;
    k *= 0x94D049BB133111EBULL;
    k ^= k >> 31;
    return k;
}

}  // namespace

bool GraphSlot::occupied() const { return depth8 != 0; }

uint8_t GraphSlot::relative_age(uint8_t generation8) const {
    return (GENERATION_CYCLE + generation8 - genPv8) & GENERATION_MASK;
}

GraphData GraphSlot::read() const {
    return GraphData{move16,
                     Value(lower16),
                     Value(upper16),
                     Value(eval16),
                     Depth(depth8 + DEPTH_ENTRY_OFFSET),
                     bool(genPv8 & 0x1)};
}

void GraphSlot::save(
  Value lower, Value upper, bool pv, Depth d, Move m, Value ev, uint8_t generation8) {
    assert(d > DEPTH_ENTRY_OFFSET);
    assert(d < 256 + DEPTH_ENTRY_OFFSET);

    if (m)
        move16 = m;

    lower16 = int16_t(clamp_to_i16(lower));
    upper16 = int16_t(clamp_to_i16(upper));
    eval16  = int16_t(clamp_to_i16(ev));
    depth8  = uint8_t(d - DEPTH_ENTRY_OFFSET);
    genPv8  = uint8_t(generation8 | uint8_t(pv));
}

void GraphEntry::reset(Key k) {
    key  = k;
    used = 1;
    for (GraphSlot& slot : slots)
        slot = GraphSlot{};
}

void RepPseudoEntry::reset(Key k, Key base, const RepContext& ctx, RepetitionKind repKind) {
    key     = k;
    baseKey = base;
    pathSig = ctx.pathSig;
    rule50  = ctx.rule50;
    kind    = repKind;
    used    = 1;
    slot    = GraphSlot{};
}

void InflightCell::clear() {
    key.store(0, std::memory_order_relaxed);
    depth.store(0, std::memory_order_relaxed);
    owner.store(0, std::memory_order_relaxed);
    waiters.store(0, std::memory_order_relaxed);
    sequence.store(0, std::memory_order_relaxed);
    state.store(uint8_t(GraphNodeState::Idle), std::memory_order_relaxed);
}

InflightGuard::InflightGuard() = default;

InflightGuard::InflightGuard(GraphTable* graph, size_t s, Key k, Depth d, uint16_t o) :
    graphTable(graph),
    slot(s),
    key(k),
    depth(int16_t(d)),
    owner(o) {}

InflightGuard::InflightGuard(InflightGuard&& other) noexcept :
    graphTable(other.graphTable),
    slot(other.slot),
    key(other.key),
    depth(other.depth),
    owner(other.owner) {
    other.graphTable = nullptr;
}

InflightGuard& InflightGuard::operator=(InflightGuard&& other) noexcept {
    if (this != &other)
    {
        release();
        graphTable       = other.graphTable;
        slot             = other.slot;
        key              = other.key;
        depth            = other.depth;
        owner            = other.owner;
        other.graphTable = nullptr;
    }

    return *this;
}

InflightGuard::~InflightGuard() { release(); }

bool InflightGuard::active() const { return graphTable != nullptr; }

void InflightGuard::release() {
    if (graphTable)
    {
        graphTable->release_inflight(slot, key, depth, owner);
        graphTable = nullptr;
    }
}

void GraphTable::resize(size_t mbSize, ThreadPool&) {
    generation8 = 0;
    reset_stats();

    if (mbSize == 0)
    {
        entries.clear();
        repEntries.clear();
        inflightEntries.reset();
        inflightEntryCount = 0;
        entries.shrink_to_fit();
        repEntries.shrink_to_fit();
        return;
    }

    const size_t bytes = mbSize * 1024 * 1024;
    entries.resize(std::max<size_t>(1, bytes / sizeof(GraphEntry)));
    repEntries.resize(std::max<size_t>(1, entries.size() / 8));
    inflightEntryCount = entries.size();
    inflightEntries.reset(new InflightCell[inflightEntryCount]);
    std::fill(entries.begin(), entries.end(), GraphEntry{});
    std::fill(repEntries.begin(), repEntries.end(), RepPseudoEntry{});
    for (size_t i = 0; i < inflightEntryCount; ++i)
        inflightEntries[i].clear();
}

void GraphTable::clear(ThreadPool&) {
    generation8 = 0;
    reset_stats();
    std::fill(entries.begin(), entries.end(), GraphEntry{});
    std::fill(repEntries.begin(), repEntries.end(), RepPseudoEntry{});
    for (size_t i = 0; i < inflightEntryCount; ++i)
        inflightEntries[i].clear();
}

void GraphTable::new_search() {
    generation8 += GENERATION_DELTA;
    reset_stats();
}

uint8_t GraphTable::generation() const { return generation8; }

bool GraphTable::enabled() const { return !entries.empty(); }

std::tuple<bool, GraphData> GraphTable::probe(Key key, Depth depth) const {
    canonicalProbes.fetch_add(1, std::memory_order_relaxed);

    GraphEntry* entry = first_entry(key);

    if (!entry)
        return {false, GraphData{Move::none(), GRAPH_LOWER_UNKNOWN, GRAPH_UPPER_UNKNOWN,
                                 VALUE_NONE, DEPTH_ENTRY_OFFSET, false}};

    if (entry->used && entry->key == key)
    {
        const GraphSlot* best = nullptr;
        for (const GraphSlot& slot : entry->slots)
            if (slot.occupied() && slot.depth8 + DEPTH_ENTRY_OFFSET >= depth
                && (!best || slot.depth8 > best->depth8))
                best = &slot;

        if (best)
        {
            canonicalHits.fetch_add(1, std::memory_order_relaxed);
            return {true, best->read()};
        }
    }

    return {false, GraphData{Move::none(), GRAPH_LOWER_UNKNOWN, GRAPH_UPPER_UNKNOWN,
                             VALUE_NONE, DEPTH_ENTRY_OFFSET, false}};
}

std::tuple<bool, GraphData> GraphTable::probe_repetition(Key              baseKey,
                                                         Depth            depth,
                                                         const RepContext& ctx,
                                                         RepetitionKind    kind) const {
    pseudoProbes.fetch_add(1, std::memory_order_relaxed);

    RepPseudoEntry* entry = first_rep_entry(repetition_key(baseKey, ctx, kind));
    if (!entry || !entry->used || entry->baseKey != baseKey || entry->kind != kind)
        return {false, GraphData{Move::none(), GRAPH_LOWER_UNKNOWN, GRAPH_UPPER_UNKNOWN,
                                 VALUE_NONE, DEPTH_ENTRY_OFFSET, false}};

    if (entry->pathSig != ctx.pathSig || entry->rule50 != ctx.rule50)
    {
        incompatiblePathSkips.fetch_add(1, std::memory_order_relaxed);
        return {false, GraphData{Move::none(), GRAPH_LOWER_UNKNOWN, GRAPH_UPPER_UNKNOWN,
                                 VALUE_NONE, DEPTH_ENTRY_OFFSET, false}};
    }

    if (!entry->slot.occupied() || entry->slot.depth8 + DEPTH_ENTRY_OFFSET < depth)
        return {false, GraphData{Move::none(), GRAPH_LOWER_UNKNOWN, GRAPH_UPPER_UNKNOWN,
                                 VALUE_NONE, DEPTH_ENTRY_OFFSET, false}};

    pseudoHits.fetch_add(1, std::memory_order_relaxed);
    return {true, entry->slot.read()};
}

void GraphTable::store_verified(Key           key,
                                Value         v,
                                bool          pv,
                                Bound         b,
                                Depth         d,
                                Move          m,
                                Value         ev,
                                uint8_t       gen,
                                PublishSource) {
    if (entries.empty() || key == 0 || b == BOUND_NONE || !is_valid(v) || d <= DEPTH_ENTRY_OFFSET)
        return;

    GraphEntry* entry = first_entry(key);
    if (!entry)
        return;

    save_slot(*entry, key, v, pv, b, d, m, ev, gen);
    verifiedStores.fetch_add(1, std::memory_order_relaxed);
    canonicalStores.fetch_add(1, std::memory_order_relaxed);
    complete_inflight(key, d, GraphNodeState::Solved);
}

void GraphTable::store_repetition(Key              baseKey,
                                  Value            v,
                                  bool             pv,
                                  Bound            b,
                                  Depth            d,
                                  Move             m,
                                  Value            ev,
                                  const RepContext& ctx,
                                  RepetitionKind    kind) {
    if (repEntries.empty() || b == BOUND_NONE || !is_valid(v) || d <= DEPTH_ENTRY_OFFSET)
        return;

    Key             k     = repetition_key(baseKey, ctx, kind);
    RepPseudoEntry* entry = first_rep_entry(k);
    if (!entry)
        return;

    if (!entry->used || entry->key != k || entry->baseKey != baseKey || entry->pathSig != ctx.pathSig
        || entry->rule50 != ctx.rule50 || entry->kind != kind)
        entry->reset(k, baseKey, ctx, kind);

    auto [lower, upper] = interval_from_bound(v, b);
    if (entry->slot.occupied() && entry->slot.depth8 == uint8_t(d - DEPTH_ENTRY_OFFSET))
    {
        GraphData prior = entry->slot.read();
        Value     mergedLower = std::max(lower, prior.lower);
        Value     mergedUpper = std::min(upper, prior.upper);
        if (mergedLower <= mergedUpper)
        {
            lower = mergedLower;
            upper = mergedUpper;
        }
    }

    entry->slot.save(lower, upper, pv, d, m, ev, generation8);
    pseudoStores.fetch_add(1, std::memory_order_relaxed);
}

void GraphTable::store_local_hint(Key, Value v, Bound b, Depth, Move, HintKind) {
    localHints.fetch_add(1, std::memory_order_relaxed);

    if (b != BOUND_NONE && is_valid(v))
        rejectedSpeculative.fetch_add(1, std::memory_order_relaxed);
}

InflightGuard GraphTable::begin_inflight(Key key, Depth depth, uint16_t owner, uint16_t preferredOwner) {
    if (!inflightEntries || inflightEntryCount == 0 || key == 0 || owner == 0)
        return {};

    const size_t slot = key % inflightEntryCount;
    InflightCell& cell = inflightEntries[slot];

    Key empty = 0;
    if (cell.key.compare_exchange_strong(empty, key, std::memory_order_acq_rel,
                                         std::memory_order_relaxed))
    {
        const bool localOwner = preferredOwner == 0 || preferredOwner == owner;
        cell.depth.store(int16_t(depth), std::memory_order_release);
        cell.owner.store(owner, std::memory_order_release);
        cell.waiters.store(0, std::memory_order_relaxed);
        cell.state.store(uint8_t(GraphNodeState::Solving), std::memory_order_release);
        inflightClaims.fetch_add(1, std::memory_order_relaxed);
        (localOwner ? localOwnerClaims : remoteOwnerClaims).fetch_add(1, std::memory_order_relaxed);
        return InflightGuard(this, slot, key, depth, owner);
    }

    inflightFallbacks.fetch_add(1, std::memory_order_relaxed);
    return {};
}

std::tuple<bool, GraphData> GraphTable::wait_for_inflight(
  Key key, Depth depth, uint16_t owner, Value, Value, int spins) const {
    InflightCell* cell = first_inflight(key);

    if (!cell)
        return {false, GraphData{Move::none(), GRAPH_LOWER_UNKNOWN, GRAPH_UPPER_UNKNOWN,
                                 VALUE_NONE, DEPTH_ENTRY_OFFSET, false}};

    const Key activeKey = cell->key.load(std::memory_order_acquire);
    if (activeKey != key)
        return {false, GraphData{Move::none(), GRAPH_LOWER_UNKNOWN, GRAPH_UPPER_UNKNOWN,
                                 VALUE_NONE, DEPTH_ENTRY_OFFSET, false}};

    const int16_t activeDepth = cell->depth.load(std::memory_order_acquire);
    const uint16_t activeOwner = cell->owner.load(std::memory_order_acquire);

    if (activeDepth < depth || activeOwner == owner)
        return {false, GraphData{Move::none(), GRAPH_LOWER_UNKNOWN, GRAPH_UPPER_UNKNOWN,
                                 VALUE_NONE, DEPTH_ENTRY_OFFSET, false}};

    inflightWaits.fetch_add(1, std::memory_order_relaxed);
    cell->waiters.fetch_add(1, std::memory_order_relaxed);
    const uint32_t observedSequence = cell->sequence.load(std::memory_order_acquire);

    for (int i = 0; i < spins; ++i)
    {
        const uint8_t state = cell->state.load(std::memory_order_acquire);
        if (cell->sequence.load(std::memory_order_acquire) != observedSequence
            || state == uint8_t(GraphNodeState::Solved)
            || state == uint8_t(GraphNodeState::Abandoned)
            || cell->key.load(std::memory_order_acquire) != key)
        {
            auto [hit, data] = probe(key, depth);
            if (hit)
            {
                inflightWaitHits.fetch_add(1, std::memory_order_relaxed);
                return {true, data};
            }

            inflightFallbacks.fetch_add(1, std::memory_order_relaxed);
            return {false, GraphData{Move::none(), GRAPH_LOWER_UNKNOWN, GRAPH_UPPER_UNKNOWN,
                                     VALUE_NONE, DEPTH_ENTRY_OFFSET, false}};
        }

        std::this_thread::yield();
    }

    inflightFallbacks.fetch_add(1, std::memory_order_relaxed);
    return {false, GraphData{Move::none(), GRAPH_LOWER_UNKNOWN, GRAPH_UPPER_UNKNOWN,
                             VALUE_NONE, DEPTH_ENTRY_OFFSET, false}};
}

uint16_t GraphTable::owner_for(Key key, size_t threadCount) const {
    if (threadCount == 0)
        return 0;

    return uint16_t((mix_key(key) % threadCount) + 1);
}

void GraphTable::record_order_probe(bool hit, bool cutoff) {
    graphOrderProbes.fetch_add(1, std::memory_order_relaxed);
    if (hit)
        graphOrderHits.fetch_add(1, std::memory_order_relaxed);
    if (cutoff)
        graphOrderCutoffs.fetch_add(1, std::memory_order_relaxed);
}

void GraphTable::record_cutoff() {
    graphCutoffs.fetch_add(1, std::memory_order_relaxed);
}

void GraphTable::record_blocked(RepetitionKind kind) {
    if (kind == RepetitionKind::HighRule50)
        blockedHighRule50.fetch_add(1, std::memory_order_relaxed);
    else if (kind != RepetitionKind::None)
        blockedRepetition.fetch_add(1, std::memory_order_relaxed);
}

GraphStats GraphTable::stats() const {
    return GraphStats{canonicalProbes.load(std::memory_order_relaxed),
                      canonicalHits.load(std::memory_order_relaxed),
                      canonicalStores.load(std::memory_order_relaxed),
                      graphCutoffs.load(std::memory_order_relaxed),
                      blockedRepetition.load(std::memory_order_relaxed),
                      blockedHighRule50.load(std::memory_order_relaxed),
                      pseudoProbes.load(std::memory_order_relaxed),
                      pseudoHits.load(std::memory_order_relaxed),
                      pseudoStores.load(std::memory_order_relaxed),
                      incompatiblePathSkips.load(std::memory_order_relaxed),
                      inflightClaims.load(std::memory_order_relaxed),
                      inflightReleases.load(std::memory_order_relaxed),
                      inflightWaits.load(std::memory_order_relaxed),
                      inflightWaitHits.load(std::memory_order_relaxed),
                      inflightFallbacks.load(std::memory_order_relaxed),
                      inflightWakeups.load(std::memory_order_relaxed),
                      inflightAbandoned.load(std::memory_order_relaxed),
                      verifiedStores.load(std::memory_order_relaxed),
                      localHints.load(std::memory_order_relaxed),
                      rejectedSpeculative.load(std::memory_order_relaxed),
                      graphOrderProbes.load(std::memory_order_relaxed),
                      graphOrderHits.load(std::memory_order_relaxed),
                      graphOrderCutoffs.load(std::memory_order_relaxed),
                      localOwnerClaims.load(std::memory_order_relaxed),
                      remoteOwnerClaims.load(std::memory_order_relaxed)};
}

std::string GraphTable::stats_string() const {
    const GraphStats s = stats();
    std::ostringstream os;
    os << "Daggerfish stats"
       << " probes " << s.canonicalProbes
       << " hits " << s.canonicalHits
       << " stores " << s.canonicalStores
       << " cutoffs " << s.graphCutoffs
       << " blocked_rep " << s.blockedRepetition
       << " blocked_rule50 " << s.blockedHighRule50
       << " pseudo_probes " << s.pseudoProbes
       << " pseudo_hits " << s.pseudoHits
       << " pseudo_stores " << s.pseudoStores
       << " path_skips " << s.incompatiblePathSkips
       << " inflight_claims " << s.inflightClaims
       << " inflight_releases " << s.inflightReleases
       << " inflight_waits " << s.inflightWaits
       << " inflight_wait_hits " << s.inflightWaitHits
       << " inflight_fallbacks " << s.inflightFallbacks
       << " inflight_wakeups " << s.inflightWakeups
       << " inflight_abandoned " << s.inflightAbandoned
       << " verified_stores " << s.verifiedStores
       << " local_hints " << s.localHints
       << " rejected_speculative " << s.rejectedSpeculative
       << " graph_order_probes " << s.graphOrderProbes
       << " graph_order_hits " << s.graphOrderHits
       << " graph_order_cutoffs " << s.graphOrderCutoffs
       << " local_owner_claims " << s.localOwnerClaims
       << " remote_owner_claims " << s.remoteOwnerClaims;
    return os.str();
}

int GraphTable::hashfull(int maxAge) const {
    if (entries.empty())
        return 0;

    const int    maxAgeInternal = maxAge << GENERATION_BITS;
    const size_t sample         = std::min<size_t>(1000, entries.size());
    int          count          = 0;

    for (size_t i = 0; i < sample; ++i)
        if (entries[i].used)
            for (const GraphSlot& slot : entries[i].slots)
                count += slot.occupied() && slot.relative_age(generation8) <= maxAgeInternal;

    return count / GraphSlotCount;
}

GraphEntry* GraphTable::first_entry(Key key) const {
    if (entries.empty())
        return nullptr;

    return const_cast<GraphEntry*>(&entries[key % entries.size()]);
}

RepPseudoEntry* GraphTable::first_rep_entry(Key key) const {
    if (repEntries.empty())
        return nullptr;

    return const_cast<RepPseudoEntry*>(&repEntries[key % repEntries.size()]);
}

InflightCell* GraphTable::first_inflight(Key key) const {
    if (!inflightEntries || inflightEntryCount == 0)
        return nullptr;

    return &inflightEntries[key % inflightEntryCount];
}

void GraphTable::save_slot(
  GraphEntry& entry, Key key, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t gen) {
    auto [lower, upper] = interval_from_bound(v, b);
    const Value incomingLower = lower;
    const Value incomingUpper = upper;

    if (!entry.used || entry.key != key)
        entry.reset(key);

    GraphSlot* replace = &entry.slots[0];

    for (GraphSlot& slot : entry.slots)
    {
        if (slot.occupied() && slot.depth8 == uint8_t(d - DEPTH_ENTRY_OFFSET))
        {
            GraphData prior = slot.read();
            lower           = std::max(lower, prior.lower);
            upper           = std::min(upper, prior.upper);
            replace         = &slot;
            break;
        }

        if (!slot.occupied())
        {
            replace = &slot;
            break;
        }

        if (replace->depth8 - replace->relative_age(gen) > slot.depth8 - slot.relative_age(gen))
            replace = &slot;
    }

    if (lower > upper)
    {
        lower = incomingLower;
        upper = incomingUpper;
    }

    replace->save(lower, upper, pv, d, m, ev, gen);
}

void GraphTable::complete_inflight(Key key, Depth depth, GraphNodeState state) {
    InflightCell* cell = first_inflight(key);
    if (!cell || cell->key.load(std::memory_order_acquire) != key
        || depth < cell->depth.load(std::memory_order_acquire))
        return;

    cell->state.store(uint8_t(state), std::memory_order_release);
    cell->sequence.fetch_add(1, std::memory_order_acq_rel);

    if (cell->waiters.load(std::memory_order_relaxed))
        inflightWakeups.fetch_add(1, std::memory_order_relaxed);

    if (state == GraphNodeState::Abandoned)
        inflightAbandoned.fetch_add(1, std::memory_order_relaxed);
}

void GraphTable::release_inflight(size_t slot, Key key, int16_t depth, uint16_t owner) {
    if (!inflightEntries || slot >= inflightEntryCount)
        return;

    InflightCell& cell = inflightEntries[slot];
    if (cell.key.load(std::memory_order_acquire) == key
        && cell.depth.load(std::memory_order_acquire) == depth
        && cell.owner.load(std::memory_order_acquire) == owner)
    {
        if (cell.state.load(std::memory_order_acquire) == uint8_t(GraphNodeState::Solving))
            complete_inflight(key, Depth(depth), GraphNodeState::Abandoned);

        cell.owner.store(0, std::memory_order_release);
        cell.depth.store(0, std::memory_order_release);
        cell.key.store(0, std::memory_order_release);
        cell.waiters.store(0, std::memory_order_relaxed);
        cell.state.store(uint8_t(GraphNodeState::Idle), std::memory_order_release);
        inflightReleases.fetch_add(1, std::memory_order_relaxed);
    }
}

void GraphTable::reset_stats() {
    canonicalProbes.store(0, std::memory_order_relaxed);
    canonicalHits.store(0, std::memory_order_relaxed);
    canonicalStores.store(0, std::memory_order_relaxed);
    graphCutoffs.store(0, std::memory_order_relaxed);
    blockedRepetition.store(0, std::memory_order_relaxed);
    blockedHighRule50.store(0, std::memory_order_relaxed);
    pseudoProbes.store(0, std::memory_order_relaxed);
    pseudoHits.store(0, std::memory_order_relaxed);
    pseudoStores.store(0, std::memory_order_relaxed);
    incompatiblePathSkips.store(0, std::memory_order_relaxed);
    inflightClaims.store(0, std::memory_order_relaxed);
    inflightReleases.store(0, std::memory_order_relaxed);
    inflightWaits.store(0, std::memory_order_relaxed);
    inflightWaitHits.store(0, std::memory_order_relaxed);
    inflightFallbacks.store(0, std::memory_order_relaxed);
    inflightWakeups.store(0, std::memory_order_relaxed);
    inflightAbandoned.store(0, std::memory_order_relaxed);
    verifiedStores.store(0, std::memory_order_relaxed);
    localHints.store(0, std::memory_order_relaxed);
    rejectedSpeculative.store(0, std::memory_order_relaxed);
    graphOrderProbes.store(0, std::memory_order_relaxed);
    graphOrderHits.store(0, std::memory_order_relaxed);
    graphOrderCutoffs.store(0, std::memory_order_relaxed);
    localOwnerClaims.store(0, std::memory_order_relaxed);
    remoteOwnerClaims.store(0, std::memory_order_relaxed);
}

RepContext make_rep_context(const Position& pos) {
    RepContext       ctx;
    const StateInfo* st = pos.state();

    ctx.rule50 = uint16_t(std::min(st->rule50, 0xFFFF));

    const int maxPly = std::min(st->rule50, st->pliesFromNull);
    for (int ply = 0; st && ply <= maxPly; ++ply)
    {
        if (ctx.count < RepContextKeyCount)
            ctx.keys[ctx.count++] = st->key;
        else
            ctx.overflow++;

        ctx.pathSig ^= mix_key(st->key + REP_HASH_SEED + Key(ply) * 0xD1B54A32D192ED03ULL);
        st = st->previous;
    }

    ctx.pathSig ^= mix_key(Key(ctx.rule50) << 32 | ctx.count);
    ctx.pathSig ^= mix_key(Key(ctx.overflow) + 0xA5A5A5A5A5A5A5A5ULL);
    return ctx;
}

Key repetition_key(Key baseKey, const RepContext& ctx, RepetitionKind kind) {
    return mix_key(baseKey ^ ctx.pathSig ^ (Key(ctx.rule50) << 32) ^ Key(kind) * REP_HASH_SEED);
}

}  // namespace Stockfish::Daggerfish
