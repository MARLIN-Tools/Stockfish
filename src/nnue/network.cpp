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

#include "network.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <type_traits>
#include <vector>

#define INCBIN_SILENCE_BITCODE_WARNING
#include "../incbin/incbin.h"

#include "../evaluate.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "nnue_misc.h"
#include "simd.h"

// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEmbeddedNNUEData[];  // a pointer to the embedded data
//     const unsigned char *const gEmbeddedNNUEEnd;     // a marker to the end
//     const unsigned int         gEmbeddedNNUESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.
#if !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
INCBIN(EmbeddedNNUEBig, EvalFileDefaultNameBig);
INCBIN(EmbeddedNNUESmall, EvalFileDefaultNameSmall);
#else
const unsigned char        gEmbeddedNNUEBigData[1]   = {0x0};
const unsigned char* const gEmbeddedNNUEBigEnd       = &gEmbeddedNNUEBigData[1];
const unsigned int         gEmbeddedNNUEBigSize      = 1;
const unsigned char        gEmbeddedNNUESmallData[1] = {0x0};
const unsigned char* const gEmbeddedNNUESmallEnd     = &gEmbeddedNNUESmallData[1];
const unsigned int         gEmbeddedNNUESmallSize    = 1;
#endif

namespace {

using namespace Stockfish::Eval::NNUE;

#if defined(USE_AVX2)
using Vec256 = __m256i;

constexpr int RecklessI16Lanes     = 16;
constexpr int RecklessBlockVectors = 8;
constexpr int RecklessBlockSize    = RecklessI16Lanes * RecklessBlockVectors;

inline Vec256 load_i16(const BiasType* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const Vec256*>(ptr));
}

inline void store_i16(BiasType* ptr, Vec256 value) {
    _mm256_storeu_si256(reinterpret_cast<Vec256*>(ptr), value);
}

inline Vec256 load_i8_to_i16(const std::int8_t* ptr) {
    return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
}

template<typename PieceWeights>
inline void add_piece_rows_avx2(
  std::array<BiasType, TransformedFeatureDimensionsBig>&              out,
  const PSQFeatureSet::IndexList&                                     features,
  const PieceWeights&                                                 pieceWeights) {
    for (IndexType offset = 0; offset < TransformedFeatureDimensionsBig; offset += RecklessBlockSize)
    {
        Vec256 regs[RecklessBlockVectors];
        for (int i = 0; i < RecklessBlockVectors; ++i)
            regs[i] = load_i16(out.data() + offset + i * RecklessI16Lanes);

        for (const auto feature : features)
        {
            const auto* row = pieceWeights.data[feature].data() + offset;
            for (int i = 0; i < RecklessBlockVectors; ++i)
                regs[i] = _mm256_add_epi16(regs[i], load_i16(row + i * RecklessI16Lanes));
        }

        for (int i = 0; i < RecklessBlockVectors; ++i)
            store_i16(out.data() + offset + i * RecklessI16Lanes, regs[i]);
    }
}

template<typename ThreatWeights>
inline void add_threat_rows_avx2(
  std::array<BiasType, TransformedFeatureDimensionsBig>&              out,
  const ThreatFeatureSet::IndexList&                                  features,
  const ThreatWeights&                                                threatWeights) {
    for (IndexType offset = 0; offset < TransformedFeatureDimensionsBig; offset += RecklessBlockSize)
    {
        Vec256 regs[RecklessBlockVectors];
        for (int i = 0; i < RecklessBlockVectors; ++i)
            regs[i] = load_i16(out.data() + offset + i * RecklessI16Lanes);

        const auto* it  = features.begin();
        const auto* end = features.end();

        while (it + 1 < end)
        {
            const auto* row0 = threatWeights.data[*it].data() + offset;
            const auto* row1 = threatWeights.data[*(it + 1)].data() + offset;

            for (int i = 0; i < RecklessBlockVectors; ++i)
            {
                const auto add0 = load_i8_to_i16(row0 + i * RecklessI16Lanes);
                const auto add1 = load_i8_to_i16(row1 + i * RecklessI16Lanes);
                regs[i]         = _mm256_add_epi16(regs[i], _mm256_add_epi16(add0, add1));
            }

            it += 2;
        }

        while (it < end)
        {
            const auto* row = threatWeights.data[*it].data() + offset;
            for (int i = 0; i < RecklessBlockVectors; ++i)
                regs[i] = _mm256_add_epi16(regs[i], load_i8_to_i16(row + i * RecklessI16Lanes));
            ++it;
        }

        for (int i = 0; i < RecklessBlockVectors; ++i)
            store_i16(out.data() + offset + i * RecklessI16Lanes, regs[i]);
    }
}

template<typename PieceWeights>
inline void apply_piece_delta_avx2(
  const std::array<BiasType, TransformedFeatureDimensionsBig>&        from,
  std::array<BiasType, TransformedFeatureDimensionsBig>&              to,
  const PSQFeatureSet::IndexList&                                     added,
  const PSQFeatureSet::IndexList&                                     removed,
  const PieceWeights&                                                 pieceWeights) {
    for (IndexType offset = 0; offset < TransformedFeatureDimensionsBig; offset += RecklessBlockSize)
    {
        Vec256 regs[RecklessBlockVectors];
        for (int i = 0; i < RecklessBlockVectors; ++i)
            regs[i] = load_i16(from.data() + offset + i * RecklessI16Lanes);

        for (const auto feature : added)
        {
            const auto* row = pieceWeights.data[feature].data() + offset;
            for (int i = 0; i < RecklessBlockVectors; ++i)
                regs[i] = _mm256_add_epi16(regs[i], load_i16(row + i * RecklessI16Lanes));
        }

        for (const auto feature : removed)
        {
            const auto* row = pieceWeights.data[feature].data() + offset;
            for (int i = 0; i < RecklessBlockVectors; ++i)
                regs[i] = _mm256_sub_epi16(regs[i], load_i16(row + i * RecklessI16Lanes));
        }

        for (int i = 0; i < RecklessBlockVectors; ++i)
            store_i16(to.data() + offset + i * RecklessI16Lanes, regs[i]);
    }
}

template<typename ThreatWeights>
inline void apply_threat_delta_avx2(
  const std::array<BiasType, TransformedFeatureDimensionsBig>&        from,
  std::array<BiasType, TransformedFeatureDimensionsBig>&              to,
  const ThreatFeatureSet::IndexList&                                  added,
  const ThreatFeatureSet::IndexList&                                  removed,
  const ThreatWeights&                                                threatWeights) {
    for (IndexType offset = 0; offset < TransformedFeatureDimensionsBig; offset += RecklessBlockSize)
    {
        Vec256 regs[RecklessBlockVectors];
        for (int i = 0; i < RecklessBlockVectors; ++i)
            regs[i] = load_i16(from.data() + offset + i * RecklessI16Lanes);

        for (const auto feature : added)
        {
            const auto* row = threatWeights.data[feature].data() + offset;
            for (int i = 0; i < RecklessBlockVectors; ++i)
                regs[i] = _mm256_add_epi16(regs[i], load_i8_to_i16(row + i * RecklessI16Lanes));
        }

        for (const auto feature : removed)
        {
            const auto* row = threatWeights.data[feature].data() + offset;
            for (int i = 0; i < RecklessBlockVectors; ++i)
                regs[i] = _mm256_sub_epi16(regs[i], load_i8_to_i16(row + i * RecklessI16Lanes));
        }

        for (int i = 0; i < RecklessBlockVectors; ++i)
            store_i16(to.data() + offset + i * RecklessI16Lanes, regs[i]);
    }
}

inline void transform_ft_avx2(
  const std::array<BiasType, TransformedFeatureDimensionsBig>& pstInput,
  const std::array<BiasType, TransformedFeatureDimensionsBig>& threatInput,
  std::uint8_t*                                                out) {
    const auto zero = _mm256_setzero_si256();
    const auto one  = _mm256_set1_epi16(255);

    for (IndexType i = 0; i < TransformedFeatureDimensionsBig / 2; i += 2 * RecklessI16Lanes)
    {
        const auto pstLhs0 = load_i16(pstInput.data() + i);
        const auto pstLhs1 = load_i16(pstInput.data() + i + RecklessI16Lanes);
        const auto pstRhs0 = load_i16(pstInput.data() + i + TransformedFeatureDimensionsBig / 2);
        const auto pstRhs1 =
          load_i16(pstInput.data() + i + TransformedFeatureDimensionsBig / 2 + RecklessI16Lanes);

        const auto threatLhs0 = load_i16(threatInput.data() + i);
        const auto threatLhs1 = load_i16(threatInput.data() + i + RecklessI16Lanes);
        const auto threatRhs0 =
          load_i16(threatInput.data() + i + TransformedFeatureDimensionsBig / 2);
        const auto threatRhs1 =
          load_i16(threatInput.data() + i + TransformedFeatureDimensionsBig / 2 + RecklessI16Lanes);

        const auto lhs0 =
          _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(pstLhs0, threatLhs0), zero), one);
        const auto lhs1 =
          _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(pstLhs1, threatLhs1), zero), one);
        const auto rhs0 =
          _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(pstRhs0, threatRhs0), zero), one);
        const auto rhs1 =
          _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(pstRhs1, threatRhs1), zero), one);

        const auto product0 = _mm256_mulhi_epi16(_mm256_slli_epi16(lhs0, 7), rhs0);
        const auto product1 = _mm256_mulhi_epi16(_mm256_slli_epi16(lhs1, 7), rhs1);

        auto packed = _mm256_packus_epi16(product0, product1);
        packed      = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));

        _mm256_storeu_si256(reinterpret_cast<Vec256*>(out + i), packed);
    }
}
#endif


struct EmbeddedNNUE {
    EmbeddedNNUE(const unsigned char* embeddedData,
                 const unsigned char* embeddedEnd,
                 const unsigned int   embeddedSize) :
        data(embeddedData),
        end(embeddedEnd),
        size(embeddedSize) {}
    const unsigned char* data;
    const unsigned char* end;
    const unsigned int   size;
};

EmbeddedNNUE get_embedded(EmbeddedNNUEType type) {
    if (type == EmbeddedNNUEType::BIG)
        return EmbeddedNNUE(gEmbeddedNNUEBigData, gEmbeddedNNUEBigEnd, gEmbeddedNNUEBigSize);
    else
        return EmbeddedNNUE(gEmbeddedNNUESmallData, gEmbeddedNNUESmallEnd, gEmbeddedNNUESmallSize);
}

}


namespace Stockfish::Eval::NNUE {


namespace Detail {

// Read evaluation function parameters
template<typename T>
bool read_parameters(std::istream& stream, T& reference) {

    std::uint32_t header;
    header = read_little_endian<std::uint32_t>(stream);
    if (!stream || header != T::get_hash_value())
        return false;
    return reference.read_parameters(stream);
}

// Write evaluation function parameters
template<typename T>
bool write_parameters(std::ostream& stream, const T& reference) {

    write_little_endian<std::uint32_t>(stream, T::get_hash_value());
    return reference.write_parameters(stream);
}

}  // namespace Detail

template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::load(const std::string& rootDirectory, std::string evalfilePath) {
#if defined(DEFAULT_NNUE_DIRECTORY)
    std::vector<std::string> dirs = {"<internal>", "", rootDirectory,
                                     stringify(DEFAULT_NNUE_DIRECTORY)};
#else
    std::vector<std::string> dirs = {"<internal>", "", rootDirectory};
#endif

    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    for (const auto& directory : dirs)
    {
        if (std::string(evalFile.current) != evalfilePath)
        {
            if (directory != "<internal>")
            {
                load_user_net(directory, evalfilePath);
            }

            if (directory == "<internal>" && evalfilePath == std::string(evalFile.defaultName))
            {
                load_internal();
            }
        }
    }
}


template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::save(const std::optional<std::string>& filename) const {
    std::string actualFilename;
    std::string msg;

    if (filename.has_value())
        actualFilename = filename.value();
    else
    {
        if (std::string(evalFile.current) != std::string(evalFile.defaultName))
        {
            msg = "Failed to export a net. "
                  "A non-embedded net can only be saved if the filename is specified";

            sync_cout << msg << sync_endl;
            return false;
        }

        actualFilename = evalFile.defaultName;
    }

    std::ofstream stream(actualFilename, std::ios_base::binary);
    bool          saved = save(stream, evalFile.current, evalFile.netDescription);

    msg = saved ? "Network saved successfully to " + actualFilename : "Failed to export a net";

    sync_cout << msg << sync_endl;
    return saved;
}


template<typename Arch, typename Transformer>
NetworkOutput
Network<Arch, Transformer>::evaluate(const Position&                         pos,
                                     AccumulatorStack&                       accumulatorStack,
                                     AccumulatorCaches::Cache<FTDimensions>& cache) const {

    constexpr uint64_t alignment = CacheLineSize;

    alignas(alignment)
      TransformedFeatureType transformedFeatures[FeatureTransformer<FTDimensions>::BufferSize];

    ASSERT_ALIGNED(transformedFeatures, alignment);

    const int  bucket = (pos.count<ALL_PIECES>() - 1) / 4;
    const auto psqt =
      featureTransformer.transform(pos, accumulatorStack, cache, transformedFeatures, bucket);
    const auto positional = network[bucket].propagate(transformedFeatures);
    return {static_cast<Value>(psqt / OutputScale), static_cast<Value>(positional / OutputScale)};
}


template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::verify(std::string                                  evalfilePath,
                                        const std::function<void(std::string_view)>& f) const {
    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    if (std::string(evalFile.current) != evalfilePath)
    {
        if (f)
        {
            std::string msg1 =
              "Network evaluation parameters compatible with the engine must be available.";
            std::string msg2 = "The network file " + evalfilePath + " was not loaded successfully.";
            std::string msg3 = "The UCI option EvalFile might need to specify the full path, "
                               "including the directory name, to the network file.";
            std::string msg4 = "The default net can be downloaded from: "
                               "https://tests.stockfishchess.org/api/nn/"
                             + std::string(evalFile.defaultName);
            std::string msg5 = "The engine will be terminated now.";

            std::string msg = "ERROR: " + msg1 + '\n' + "ERROR: " + msg2 + '\n' + "ERROR: " + msg3
                            + '\n' + "ERROR: " + msg4 + '\n' + "ERROR: " + msg5 + '\n';

            f(msg);
        }

        exit(EXIT_FAILURE);
    }

    if (f)
    {
        size_t size = sizeof(featureTransformer) + sizeof(Arch) * LayerStacks;
        f("NNUE evaluation using " + evalfilePath + " (" + std::to_string(size / (1024 * 1024))
          + "MiB, (" + std::to_string(featureTransformer.TotalInputDimensions) + ", "
          + std::to_string(network[0].TransformedFeatureDimensions) + ", "
          + std::to_string(network[0].FC_0_OUTPUTS) + ", " + std::to_string(network[0].FC_1_OUTPUTS)
          + ", 1))");
    }
}


template<typename Arch, typename Transformer>
NnueEvalTrace
Network<Arch, Transformer>::trace_evaluate(const Position&                         pos,
                                           AccumulatorStack&                       accumulatorStack,
                                           AccumulatorCaches::Cache<FTDimensions>& cache) const {

    constexpr uint64_t alignment = CacheLineSize;

    alignas(alignment)
      TransformedFeatureType transformedFeatures[FeatureTransformer<FTDimensions>::BufferSize];

    ASSERT_ALIGNED(transformedFeatures, alignment);

    NnueEvalTrace t{};
    t.correctBucket = (pos.count<ALL_PIECES>() - 1) / 4;
    for (IndexType bucket = 0; bucket < LayerStacks; ++bucket)
    {
        const auto materialist =
          featureTransformer.transform(pos, accumulatorStack, cache, transformedFeatures, bucket);
        const auto positional = network[bucket].propagate(transformedFeatures);

        t.psqt[bucket]       = static_cast<Value>(materialist / OutputScale);
        t.positional[bucket] = static_cast<Value>(positional / OutputScale);
    }

    return t;
}


template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::load_user_net(const std::string& dir,
                                               const std::string& evalfilePath) {
    std::ifstream stream(dir + evalfilePath, std::ios::binary);
    auto          description = load(stream);

    if (description.has_value())
    {
        evalFile.current        = evalfilePath;
        evalFile.netDescription = description.value();
    }
}


template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::load_internal() {
    // C++ way to prepare a buffer for a memory stream
    class MemoryBuffer: public std::basic_streambuf<char> {
       public:
        MemoryBuffer(char* p, size_t n) {
            setg(p, p, p + n);
            setp(p, p + n);
        }
    };

    const auto embedded = get_embedded(embeddedType);

    MemoryBuffer buffer(const_cast<char*>(reinterpret_cast<const char*>(embedded.data)),
                        size_t(embedded.size));

    std::istream stream(&buffer);
    auto         description = load(stream);

    if (description.has_value())
    {
        evalFile.current        = evalFile.defaultName;
        evalFile.netDescription = description.value();
    }
}


template<typename Arch, typename Transformer>
void Network<Arch, Transformer>::initialize() {
    initialized = true;
}


template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::save(std::ostream&      stream,
                                      const std::string& name,
                                      const std::string& netDescription) const {
    if (name.empty() || name == "None")
        return false;

    return write_parameters(stream, netDescription);
}


template<typename Arch, typename Transformer>
std::optional<std::string> Network<Arch, Transformer>::load(std::istream& stream) {
    initialize();
    std::string description;

    return read_parameters(stream, description) ? std::make_optional(description) : std::nullopt;
}


template<typename Arch, typename Transformer>
std::size_t Network<Arch, Transformer>::get_content_hash() const {
    if (!initialized)
        return 0;

    std::size_t h = 0;
    hash_combine(h, featureTransformer);
    for (auto&& layerstack : network)
        hash_combine(h, layerstack);
    hash_combine(h, evalFile);
    hash_combine(h, static_cast<int>(embeddedType));
    return h;
}

// Read network header
template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::read_header(std::istream&  stream,
                                             std::uint32_t* hashValue,
                                             std::string*   desc) const {
    std::uint32_t version, size;

    version    = read_little_endian<std::uint32_t>(stream);
    *hashValue = read_little_endian<std::uint32_t>(stream);
    size       = read_little_endian<std::uint32_t>(stream);
    if (!stream || version != Version)
        return false;
    desc->resize(size);
    stream.read(&(*desc)[0], size);
    return !stream.fail();
}


// Write network header
template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::write_header(std::ostream&      stream,
                                              std::uint32_t      hashValue,
                                              const std::string& desc) const {
    write_little_endian<std::uint32_t>(stream, Version);
    write_little_endian<std::uint32_t>(stream, hashValue);
    write_little_endian<std::uint32_t>(stream, std::uint32_t(desc.size()));
    stream.write(&desc[0], desc.size());
    return !stream.fail();
}


template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::read_parameters(std::istream& stream,
                                                 std::string&  netDescription) {
    std::uint32_t hashValue;
    if (!read_header(stream, &hashValue, &netDescription))
        return false;
    if (hashValue != Network::hash)
        return false;
    if (!Detail::read_parameters(stream, featureTransformer))
        return false;
    for (std::size_t i = 0; i < LayerStacks; ++i)
    {
        if (!Detail::read_parameters(stream, network[i]))
            return false;
    }
    return stream && stream.peek() == std::ios::traits_type::eof();
}


template<typename Arch, typename Transformer>
bool Network<Arch, Transformer>::write_parameters(std::ostream&      stream,
                                                  const std::string& netDescription) const {
    if (!write_header(stream, Network::hash, netDescription))
        return false;
    if (!Detail::write_parameters(stream, featureTransformer))
        return false;
    for (std::size_t i = 0; i < LayerStacks; ++i)
    {
        if (!Detail::write_parameters(stream, network[i]))
            return false;
    }
    return bool(stream);
}

void RecklessRawBigNetwork::load(const std::string& rootDirectory, std::string evalfilePath) {
    const auto join_path = [](const std::string& root, const std::string& file) {
        if (root.empty())
            return file;
        const char sep = root.back() == '/' || root.back() == '\\' ? '\0' : '/';
        return sep ? root + sep + file : root + file;
    };

    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    initialized             = false;
    evalFile.current        = "None";
    evalFile.netDescription = "";

    std::vector<std::string> candidates = {evalfilePath};
    if (!rootDirectory.empty())
        candidates.push_back(join_path(rootDirectory, evalfilePath));

    for (const auto& candidate : candidates)
    {
        if (load_reckless_file(candidate))
        {
            evalFile.current        = evalfilePath;
            evalFile.netDescription = "Reckless raw network";
            return;
        }
    }

    if (evalfilePath == std::string(evalFile.defaultName)
        && try_download_default(rootDirectory, evalfilePath))
    {
        for (const auto& candidate : candidates)
        {
            if (load_reckless_file(candidate))
            {
                evalFile.current        = evalfilePath;
                evalFile.netDescription = "Reckless raw network";
                return;
            }
        }
    }
}

bool RecklessRawBigNetwork::save(const std::optional<std::string>& filename) const {
    if (!initialized)
        return false;

    if (!filename.has_value())
        return false;

    std::ofstream stream(filename.value(), std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&parameters), sizeof(parameters));
    return bool(stream);
}

std::size_t RecklessRawBigNetwork::get_content_hash() const {
    if (!initialized)
        return 0;

    std::size_t h = 0;
    hash_combine(h, evalFile);

    const auto* bytes = reinterpret_cast<const unsigned char*>(&parameters);
    std::size_t raw   = 1469598103934665603ull;
    for (std::size_t i = 0; i < sizeof(parameters); i += 4096)
    {
        raw ^= bytes[i];
        raw *= 1099511628211ull;
    }
    raw ^= bytes[sizeof(parameters) - 1];
    raw *= 1099511628211ull;

    hash_combine(h, raw);
    return h;
}

NetworkOutput RecklessRawBigNetwork::evaluate(
  const Position& pos,
  AccumulatorStack& accumulatorStack,
  AccumulatorCaches::Cache<FTDimensions>&) const {
    if (!initialized)
        return {VALUE_ZERO, VALUE_ZERO};

    ensure_pst(WHITE, pos, accumulatorStack);
    ensure_threats(WHITE, pos, accumulatorStack);
    ensure_pst(BLACK, pos, accumulatorStack);
    ensure_threats(BLACK, pos, accumulatorStack);

    FtArray          ftOut{};
    NnzArray         nnz{};
    const std::size_t bucket = OutputBucketsLayout[pos.count<ALL_PIECES>()];
    transform(pos, accumulatorStack.latest_reckless_raw(), ftOut);
    const std::size_t nnzCount = find_nnz(ftOut, nnz);
    return {VALUE_ZERO, evaluate_bucket(ftOut, nnz, nnzCount, bucket)};
}

bool RecklessRawBigNetwork::load_reckless_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        return false;

    if (static_cast<std::size_t>(stream.tellg()) != sizeof(parameters))
        return false;

    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(&parameters), sizeof(parameters));
    if (!stream)
        return false;

    initialized = true;
    return true;
}

bool RecklessRawBigNetwork::try_download_default(const std::string& rootDirectory,
                                                 const std::string& filename) {
    constexpr auto Url =
      "https://github.com/codedeliveryservice/RecklessNetworks/releases/download/networks/"
      "v58-ca025eef.nnue";

    const std::string out =
      rootDirectory.empty() ? filename
                            : (rootDirectory.back() == '/' || rootDirectory.back() == '\\'
                                 ? rootDirectory + filename
                                 : rootDirectory + '/' + filename);
    const std::string command =
      "curl -fsSL -o \"" + out + "\" \"" + std::string(Url) + "\"";
    return std::system(command.c_str()) == 0;
}

void RecklessRawBigNetwork::refresh_pst(const Position&                    pos,
                                        Color                              pov,
                                        std::array<BiasType, FTDimensions>& out) const {
    out = parameters.ft_biases.data;

    PSQFeatureSet::IndexList pieceFeatures;
    PSQFeatureSet::append_active_indices(pov, pos, pieceFeatures);

#if defined(USE_AVX2)
    add_piece_rows_avx2(out, pieceFeatures, parameters.ft_piece_weights);
#else
    for (const auto index : pieceFeatures)
        for (IndexType i = 0; i < FTDimensions; ++i)
            out[i] += parameters.ft_piece_weights.data[index][i];
#endif
}

void RecklessRawBigNetwork::refresh_threats(
  const Position& pos, Color pov, std::array<BiasType, FTDimensions>& out) const {
    out.fill(0);

    ThreatFeatureSet::IndexList threatFeatures;
    ThreatFeatureSet::append_active_indices(pov, pos, threatFeatures);

#if defined(USE_AVX2)
    add_threat_rows_avx2(out, threatFeatures, parameters.ft_threat_weights);
#else
    for (const auto index : threatFeatures)
        for (IndexType i = 0; i < FTDimensions; ++i)
            out[i] += parameters.ft_threat_weights.data[index][i];
#endif
}

void RecklessRawBigNetwork::update_pst_incremental(
  Color                                          pov,
  Square                                         ksq,
  const DirtyPiece&                              diff,
  const std::array<BiasType, FTDimensions>&      from,
  std::array<BiasType, FTDimensions>&            to) const {
    PSQFeatureSet::IndexList removed;
    PSQFeatureSet::IndexList added;
    PSQFeatureSet::append_changed_indices(pov, ksq, diff, removed, added);

#if defined(USE_AVX2)
    apply_piece_delta_avx2(from, to, added, removed, parameters.ft_piece_weights);
#else
    to = from;
    for (const auto index : added)
        for (IndexType i = 0; i < FTDimensions; ++i)
            to[i] += parameters.ft_piece_weights.data[index][i];
    for (const auto index : removed)
        for (IndexType i = 0; i < FTDimensions; ++i)
            to[i] -= parameters.ft_piece_weights.data[index][i];
#endif
}

void RecklessRawBigNetwork::update_threat_incremental(
  Color                                          pov,
  Square                                         ksq,
  const DirtyThreats&                            diff,
  const std::array<BiasType, FTDimensions>&      from,
  std::array<BiasType, FTDimensions>&            to) const {
    ThreatFeatureSet::IndexList removed;
    ThreatFeatureSet::IndexList added;
    ThreatFeatureSet::append_changed_indices(pov, ksq, diff, removed, added);

#if defined(USE_AVX2)
    apply_threat_delta_avx2(from, to, added, removed, parameters.ft_threat_weights);
#else
    to = from;
    for (const auto index : added)
        for (IndexType i = 0; i < FTDimensions; ++i)
            to[i] += parameters.ft_threat_weights.data[index][i];
    for (const auto index : removed)
        for (IndexType i = 0; i < FTDimensions; ++i)
            to[i] -= parameters.ft_threat_weights.data[index][i];
#endif
}

void RecklessRawBigNetwork::ensure_pst(Color pov,
                                       const Position& pos,
                                       AccumulatorStack& accumulatorStack) const {
    auto&       latest = accumulatorStack.mut_latest_reckless_raw();
    const auto  size   = accumulatorStack.current_size();
    const auto  ksq    = pos.square<KING>(pov);

    if (latest.pstComputed[pov])
        return;

    std::size_t base = size;
    for (std::size_t idx = size - 1; idx > 0; --idx)
    {
        if (accumulatorStack.reckless_raw_at(idx).pstComputed[pov])
        {
            base = idx;
            break;
        }

        if (PSQFeatureSet::requires_refresh(accumulatorStack.psq_diff_at(idx), pov))
            break;
    }

    if (base == size)
    {
        refresh_pst(pos, pov, latest.pst[pov]);
        latest.pstComputed[pov] = true;
        return;
    }

    for (std::size_t idx = base + 1; idx < size; ++idx)
    {
        auto&       current = accumulatorStack.mut_reckless_raw_at(idx);
        const auto& prev    = accumulatorStack.reckless_raw_at(idx - 1);
        update_pst_incremental(pov, ksq, accumulatorStack.psq_diff_at(idx), prev.pst[pov],
                               current.pst[pov]);
        current.pstComputed[pov] = true;
    }
}

void RecklessRawBigNetwork::ensure_threats(Color pov,
                                           const Position& pos,
                                           AccumulatorStack& accumulatorStack) const {
    auto&       latest = accumulatorStack.mut_latest_reckless_raw();
    const auto  size   = accumulatorStack.current_size();
    const auto  ksq    = pos.square<KING>(pov);

    if (latest.threatComputed[pov])
        return;

    std::size_t base = size;
    for (std::size_t idx = size - 1; idx > 0; --idx)
    {
        if (accumulatorStack.reckless_raw_at(idx).threatComputed[pov])
        {
            base = idx;
            break;
        }

        if (ThreatFeatureSet::requires_refresh(accumulatorStack.threat_diff_at(idx), pov))
            break;
    }

    if (base == size)
    {
        refresh_threats(pos, pov, latest.threat[pov]);
        latest.threatComputed[pov] = true;
        return;
    }

    for (std::size_t idx = base + 1; idx < size; ++idx)
    {
        auto&       current = accumulatorStack.mut_reckless_raw_at(idx);
        const auto& prev    = accumulatorStack.reckless_raw_at(idx - 1);
        update_threat_incremental(pov, ksq, accumulatorStack.threat_diff_at(idx),
                                  prev.threat[pov], current.threat[pov]);
        current.threatComputed[pov] = true;
    }
}

void RecklessRawBigNetwork::transform(const Position&               pos,
                                      const RecklessRawAccumulator& accumulators,
                                      FtArray&                      ftOut) const {
    const Color stm = pos.side_to_move();
    for (int flip = 0; flip < 2; ++flip)
    {
        const auto& pstInput    = accumulators.pst[stm ^ Color(flip)];
        const auto& threatInput = accumulators.threat[stm ^ Color(flip)];

#if defined(USE_AVX2)
        transform_ft_avx2(pstInput, threatInput, ftOut.data() + flip * FTDimensions / 2);
#else
        for (IndexType i = 0; i < FTDimensions / 2; ++i)
        {
            const int left = std::clamp<int>(pstInput[i] + threatInput[i], 0, 255);
            const int right =
              std::clamp<int>(pstInput[i + FTDimensions / 2] + threatInput[i + FTDimensions / 2],
                              0, 255);
            ftOut[i + flip * FTDimensions / 2] = static_cast<std::uint8_t>((left * right) >> 9);
        }
#endif
    }
}

std::size_t RecklessRawBigNetwork::find_nnz(const FtArray& ftOut, NnzArray& nnz) const {
    std::size_t count = 0;
    for (IndexType chunk = 0; chunk < FTDimensions / 4; ++chunk)
    {
        const std::size_t base = chunk * 4;
        if (ftOut[base] | ftOut[base + 1] | ftOut[base + 2] | ftOut[base + 3])
            nnz[count++] = static_cast<std::uint16_t>(chunk);
    }

    return count;
}

void RecklessRawBigNetwork::propagate_l1(const FtArray&            ftOut,
                                         const NnzArray&           nnz,
                                         std::size_t               nnzCount,
                                         std::size_t               bucket,
                                         std::array<float, 16>&    l1) const {
    std::array<int32_t, 16> preActivation{};
    for (std::size_t idx = 0; idx < nnzCount; ++idx)
    {
        const IndexType chunk = nnz[idx];
        const std::size_t base = chunk * 4;
        for (std::size_t neuron = 0; neuron < 16; ++neuron)
        {
            const std::size_t offset = chunk * 16 * 4 + neuron * 4;
            preActivation[neuron] +=
              ftOut[base] * parameters.l1_weights.data[bucket][offset]
              + ftOut[base + 1] * parameters.l1_weights.data[bucket][offset + 1]
              + ftOut[base + 2] * parameters.l1_weights.data[bucket][offset + 2]
              + ftOut[base + 3] * parameters.l1_weights.data[bucket][offset + 3];
        }
    }

    for (std::size_t i = 0; i < l1.size(); ++i)
        l1[i] =
          std::clamp(preActivation[i] * DequantMultiplier + parameters.l1_biases.data[bucket][i],
                     0.0f, 1.0f);
}

Value RecklessRawBigNetwork::evaluate_bucket(
  const FtArray& ftOut, const NnzArray& nnz, std::size_t nnzCount, std::size_t bucket) const {
    std::array<float, 16> l1{};
    propagate_l1(ftOut, nnz, nnzCount, bucket, l1);

    std::array<float, L3Big> l2 = parameters.l2_biases.data[bucket];
    for (std::size_t i = 0; i < l1.size(); ++i)
        for (int j = 0; j < L3Big; ++j)
            l2[j] += parameters.l2_weights.data[bucket][i][j] * l1[i];

    for (auto& v : l2)
        v = std::clamp(v, 0.0f, 1.0f);

    float output = parameters.l3_biases.data[bucket];
    for (int i = 0; i < L3Big; ++i)
        output += parameters.l3_weights.data[bucket][i] * l2[i];

    return static_cast<Value>(output * NetworkScale);
}

void RecklessRawBigNetwork::verify(
  std::string evalfilePath, const std::function<void(std::string_view)>& f) const {
    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    if (std::string(evalFile.current) != evalfilePath)
    {
        if (f)
        {
            const std::string msg =
              "ERROR: Reckless raw network parameters must be available.\n"
              "ERROR: The network file " + evalfilePath + " was not loaded successfully.\n"
              "ERROR: The file can be downloaded from:\n"
              "ERROR: https://github.com/codedeliveryservice/RecklessNetworks/releases/download/"
              "networks/v58-ca025eef.nnue\n"
              "ERROR: The engine will be terminated now.\n";
            f(msg);
        }
        exit(EXIT_FAILURE);
    }

    if (f)
    {
        const auto size = sizeof(parameters) / (1024 * 1024);
        f("NNUE evaluation using " + evalfilePath + " (" + std::to_string(size)
          + "MiB, (74544, 768, 16, 32, 1))");
    }
}

NnueEvalTrace RecklessRawBigNetwork::trace_evaluate(
  const Position& pos,
  AccumulatorStack& accumulatorStack,
  AccumulatorCaches::Cache<FTDimensions>&) const {
    NnueEvalTrace out{};
    if (!initialized)
        return out;

    ensure_pst(WHITE, pos, accumulatorStack);
    ensure_threats(WHITE, pos, accumulatorStack);
    ensure_pst(BLACK, pos, accumulatorStack);
    ensure_threats(BLACK, pos, accumulatorStack);

    FtArray ftOut{};
    NnzArray nnz{};
    transform(pos, accumulatorStack.latest_reckless_raw(), ftOut);
    const std::size_t nnzCount = find_nnz(ftOut, nnz);
    out.correctBucket = OutputBucketsLayout[pos.count<ALL_PIECES>()];
    for (std::size_t bucket = 0; bucket < LayerStacks; ++bucket)
    {
        out.psqt[bucket]       = VALUE_ZERO;
        out.positional[bucket] = evaluate_bucket(ftOut, nnz, nnzCount, bucket);
    }
    return out;
}

// Explicit template instantiations

template class Network<NetworkArchitecture<TransformedFeatureDimensionsBig, L2Big, L3Big>,
                       FeatureTransformer<TransformedFeatureDimensionsBig>>;

template class Network<NetworkArchitecture<TransformedFeatureDimensionsSmall, L2Small, L3Small>,
                       FeatureTransformer<TransformedFeatureDimensionsSmall>>;

}  // namespace Stockfish::Eval::NNUE
