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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "../misc.h"
#include "../types.h"
#include "nnue_accumulator.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "nnue_feature_transformer.h"
#include "nnue_misc.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE {

enum class EmbeddedNNUEType {
    BIG,
    SMALL,
};

using NetworkOutput = std::tuple<Value, Value>;

// The network must be a trivial type, i.e. the memory must be in-line.
// This is required to allow sharing the network via shared memory, as
// there is no way to run destructors.
template<typename Arch, typename Transformer>
class Network {
    static constexpr IndexType FTDimensions = Arch::TransformedFeatureDimensions;

   public:
    Network(EvalFile file, EmbeddedNNUEType type) :
        evalFile(file),
        embeddedType(type) {}

    Network(const Network& other) = default;
    Network(Network&& other)      = default;

    Network& operator=(const Network& other) = default;
    Network& operator=(Network&& other)      = default;

    void load(const std::string& rootDirectory, std::string evalfilePath);
    bool save(const std::optional<std::string>& filename) const;

    std::size_t get_content_hash() const;
    const auto& get_biases() const { return featureTransformer.biases; }

    NetworkOutput evaluate(const Position&                         pos,
                           AccumulatorStack&                       accumulatorStack,
                           AccumulatorCaches::Cache<FTDimensions>& cache) const;


    void verify(std::string evalfilePath, const std::function<void(std::string_view)>&) const;
    NnueEvalTrace trace_evaluate(const Position&                         pos,
                                 AccumulatorStack&                       accumulatorStack,
                                 AccumulatorCaches::Cache<FTDimensions>& cache) const;

   private:
    void load_user_net(const std::string&, const std::string&);
    void load_internal();

    void initialize();

    bool                       save(std::ostream&, const std::string&, const std::string&) const;
    std::optional<std::string> load(std::istream&);

    bool read_header(std::istream&, std::uint32_t*, std::string*) const;
    bool write_header(std::ostream&, std::uint32_t, const std::string&) const;

    bool read_parameters(std::istream&, std::string&);
    bool write_parameters(std::ostream&, const std::string&) const;

    // Input feature converter
    Transformer featureTransformer;

    // Evaluation function
    Arch network[LayerStacks];

    EvalFile         evalFile;
    EmbeddedNNUEType embeddedType;

    bool initialized = false;

    // Hash value of evaluation function structure
    static constexpr std::uint32_t hash = Transformer::get_hash_value() ^ Arch::get_hash_value();

    template<IndexType Size>
    friend struct AccumulatorCaches::Cache;
};

class RecklessRawBigNetwork {
    static constexpr IndexType FTDimensions = TransformedFeatureDimensionsBig;

    template<typename T>
    struct alignas(64) Aligned {
        T data;
    };

    struct Parameters {
        Aligned<std::array<std::array<std::int8_t, FTDimensions>, ThreatFeatureSet::Dimensions>>
          ft_threat_weights;
        Aligned<std::array<std::array<BiasType, FTDimensions>, PSQFeatureSet::Dimensions>>
          ft_piece_weights;
        Aligned<std::array<BiasType, FTDimensions>> ft_biases;
        Aligned<std::array<std::array<std::int8_t, 16 * FTDimensions>, LayerStacks>> l1_weights;
        Aligned<std::array<std::array<float, 16>, LayerStacks>>                      l1_biases;
        Aligned<std::array<std::array<std::array<float, L3Big>, 16>, LayerStacks>>   l2_weights;
        Aligned<std::array<std::array<float, L3Big>, LayerStacks>>                    l2_biases;
        Aligned<std::array<std::array<float, L3Big>, LayerStacks>>                    l3_weights;
        Aligned<std::array<float, LayerStacks>>                                       l3_biases;
    };

   public:
    RecklessRawBigNetwork(EvalFile file, EmbeddedNNUEType type) :
        evalFile(file),
        embeddedType(type) {}

    void load(const std::string& rootDirectory, std::string evalfilePath);
    bool save(const std::optional<std::string>& filename) const;

    std::size_t get_content_hash() const;
    const auto& get_biases() const { return parameters.ft_biases.data; }

    NetworkOutput evaluate(const Position&                         pos,
                           AccumulatorStack&                       accumulatorStack,
                           AccumulatorCaches::Cache<FTDimensions>& cache) const;
    void          verify(std::string evalfilePath,
                         const std::function<void(std::string_view)>&) const;
    NnueEvalTrace trace_evaluate(const Position&                         pos,
                                 AccumulatorStack&                       accumulatorStack,
                                 AccumulatorCaches::Cache<FTDimensions>& cache) const;

   private:
    static constexpr float DequantMultiplier = float(1 << 9) / float(255 * 255 * 64);
    static constexpr int   NetworkScale      = 380;
    static constexpr std::array<std::size_t, 33> OutputBucketsLayout = {
      0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1,
      2, 2, 2, 2,
      3, 3, 3,
      4, 4, 4,
      5, 5, 5,
      6, 6, 6,
      7, 7, 7, 7,
    };

    using FtArray = std::array<std::uint8_t, FTDimensions>;
    using AccArray = std::array<std::array<BiasType, FTDimensions>, COLOR_NB>;
    using NnzArray = std::array<std::uint16_t, FTDimensions / 4>;

    bool load_reckless_file(const std::string& path);
    bool try_download_default(const std::string& rootDirectory, const std::string& filename);

    void refresh_pst(const Position& pos, Color pov, std::array<BiasType, FTDimensions>& out) const;
    void refresh_threats(const Position& pos, Color pov, std::array<BiasType, FTDimensions>& out) const;
    void update_pst_incremental(Color                          pov,
                                Square                         ksq,
                                const DirtyPiece&              diff,
                                const std::array<BiasType, FTDimensions>& from,
                                std::array<BiasType, FTDimensions>&       to) const;
    void update_threat_incremental(Color                          pov,
                                   Square                         ksq,
                                   const DirtyThreats&            diff,
                                   const std::array<BiasType, FTDimensions>& from,
                                   std::array<BiasType, FTDimensions>&       to) const;
    void ensure_pst(Color pov, const Position& pos, AccumulatorStack& accumulatorStack) const;
    void ensure_threats(Color pov, const Position& pos, AccumulatorStack& accumulatorStack) const;
    void transform(const Position& pos,
                   const RecklessRawAccumulator& accumulators,
                   FtArray&                      ftOut) const;
    std::size_t find_nnz(const FtArray& ftOut, NnzArray& nnz) const;
    void propagate_l1(const FtArray& ftOut,
                      const NnzArray& nnz,
                      std::size_t     nnzCount,
                      std::size_t     bucket,
                      std::array<float, 16>& l1) const;
    Value evaluate_bucket(const FtArray& ftOut,
                          const NnzArray& nnz,
                          std::size_t     nnzCount,
                          std::size_t     bucket) const;

    Parameters       parameters{};
    EvalFile         evalFile;
    EmbeddedNNUEType embeddedType;
    bool             initialized = false;
};

// Definitions of the network types
using SmallFeatureTransformer = FeatureTransformer<TransformedFeatureDimensionsSmall>;
using SmallNetworkArchitecture =
  NetworkArchitecture<TransformedFeatureDimensionsSmall, L2Small, L3Small>;

using BigFeatureTransformer  = FeatureTransformer<TransformedFeatureDimensionsBig>;
using BigNetworkArchitecture = NetworkArchitecture<TransformedFeatureDimensionsBig, L2Big, L3Big>;

using NetworkBig   = RecklessRawBigNetwork;
using NetworkSmall = Network<SmallNetworkArchitecture, SmallFeatureTransformer>;


struct Networks {
    Networks(EvalFile bigFile, EvalFile smallFile) :
        big(bigFile, EmbeddedNNUEType::BIG),
        small(smallFile, EmbeddedNNUEType::SMALL) {}

    NetworkBig   big;
    NetworkSmall small;
};


}  // namespace Stockfish

template<typename ArchT, typename FeatureTransformerT>
struct std::hash<Stockfish::Eval::NNUE::Network<ArchT, FeatureTransformerT>> {
    std::size_t operator()(
      const Stockfish::Eval::NNUE::Network<ArchT, FeatureTransformerT>& network) const noexcept {
        return network.get_content_hash();
    }
};

template<>
struct std::hash<Stockfish::Eval::NNUE::RecklessRawBigNetwork> {
    std::size_t
    operator()(const Stockfish::Eval::NNUE::RecklessRawBigNetwork& network) const noexcept {
        return network.get_content_hash();
    }
};

template<>
struct std::hash<Stockfish::Eval::NNUE::Networks> {
    std::size_t operator()(const Stockfish::Eval::NNUE::Networks& networks) const noexcept {
        std::size_t h = 0;
        Stockfish::hash_combine(h, networks.big);
        Stockfish::hash_combine(h, networks.small);
        return h;
    }
};

#endif
