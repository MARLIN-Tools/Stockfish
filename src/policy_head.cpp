#include "policy_head.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

#if defined(USE_AVX2)
    #include <immintrin.h>
#endif

#include "position.h"

namespace Stockfish::Policy {
namespace {

struct MovePolicyWeights {
    int16_t proj[8][2 * Eval::NNUE::L2Big + 1]{};
    int16_t from[64][8]{};
    int16_t to[64][8]{};
    int16_t piece[16][8]{};
    int16_t promo[8][8]{};
    int16_t flags[8][8]{};
    int16_t bucketBias[8]{};
    int16_t nodeBias[3]{};
};

MovePolicyWeights gWeights;
bool              gEnabled = false;
bool              gLoaded  = false;
std::string       gPath;
constexpr char    PolicyMagic[] = {'X', 'A', 'N', 'P', 'O', 'L', '1', '\0'};

template<typename T>
bool read_scalar(const std::vector<char>& blob, size_t& offset, T& out) {
    if (offset + sizeof(T) > blob.size())
        return false;
    std::memcpy(&out, blob.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

bool read_array(const std::vector<char>& blob, size_t& offset, int expectedCount, int16_t* out) {
    uint32_t count = 0;
    if (!read_scalar(blob, offset, count) || count != static_cast<uint32_t>(expectedCount))
        return false;

    const size_t byteCount = size_t(count) * sizeof(int16_t);
    if (offset + byteCount > blob.size())
        return false;

    std::memcpy(out, blob.data() + offset, byteCount);
    offset += byteCount;
    return true;
}

int dot_i16(const int16_t* a, const int16_t* b, int count) {
    int sum = 0;

#if defined(USE_AVX2)
    __m256i acc = _mm256_setzero_si256();
    int     i   = 0;
    for (; i + 16 <= count; i += 16)
    {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        acc              = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }

    alignas(32) int lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
    for (int lane : lanes)
        sum += lane;

    for (; i < count; ++i)
        sum += int(a[i]) * int(b[i]);
#else
    for (int i = 0; i < count; ++i)
        sum += int(a[i]) * int(b[i]);
#endif

    return sum;
}

}  // namespace

void set_enabled(bool enabled_) { gEnabled = enabled_; }

bool enabled() { return gEnabled && gLoaded; }

bool load(const std::string& path, std::string* error) {
    if (path.empty())
    {
        reset();
        return true;
    }

    FILE* in = std::fopen(path.c_str(), "rb");
    if (!in)
    {
        reset();
        if (error)
            *error = "Failed to open policy file: " + path;
        return false;
    }

    std::fseek(in, 0, SEEK_END);
    long fileSize = std::ftell(in);
    std::rewind(in);

    if (fileSize < 0)
    {
        std::fclose(in);
        reset();
        if (error)
            *error = "Failed to stat policy file: " + path;
        return false;
    }

    std::vector<char> data(static_cast<size_t>(fileSize));
    const size_t readCount = data.empty() ? 0 : std::fread(data.data(), 1, data.size(), in);
    std::fclose(in);
    if (readCount != data.size())
    {
        reset();
        if (error)
            *error = "Failed to read policy file: " + path;
        return false;
    }

    if (data.size() < sizeof(PolicyMagic) + sizeof(uint32_t))
    {
        reset();
        if (error)
            *error = "Policy file too small: " + path;
        return false;
    }

    const auto it = std::find_end(data.begin(), data.end(), std::begin(PolicyMagic), std::end(PolicyMagic));
    if (it == data.end())
    {
        reset();
        if (error)
            *error = "Policy trailer not found: " + path;
        return false;
    }

    size_t offset = size_t(std::distance(data.begin(), it)) + sizeof(PolicyMagic);
    uint32_t payloadLen = 0;
    if (!read_scalar(data, offset, payloadLen) || offset + payloadLen > data.size())
    {
        reset();
        if (error)
            *error = "Invalid policy payload length: " + path;
        return false;
    }

    const size_t payloadEnd = offset + payloadLen;
    uint32_t version = 0, dim = 0, tapSize = 0;
    if (!read_scalar(data, offset, version) || !read_scalar(data, offset, dim)
        || !read_scalar(data, offset, tapSize) || version != 1 || dim != 8
        || tapSize != 2 * Eval::NNUE::L2Big + 1)
    {
        reset();
        if (error)
            *error = "Unsupported policy format: " + path;
        return false;
    }

    MovePolicyWeights loaded{};
    if (!read_array(data, offset, 8 * (2 * Eval::NNUE::L2Big + 1), &loaded.proj[0][0])
        || !read_array(data, offset, 64 * 8, &loaded.from[0][0])
        || !read_array(data, offset, 64 * 8, &loaded.to[0][0])
        || !read_array(data, offset, 16 * 8, &loaded.piece[0][0])
        || !read_array(data, offset, 8 * 8, &loaded.promo[0][0])
        || !read_array(data, offset, 8 * 8, &loaded.flags[0][0])
        || !read_array(data, offset, 8, &loaded.bucketBias[0])
        || !read_array(data, offset, 3, &loaded.nodeBias[0]) || offset != payloadEnd)
    {
        reset();
        if (error)
            *error = "Corrupt policy payload: " + path;
        return false;
    }

    gWeights = loaded;
    gLoaded  = true;
    gPath    = path;
    if (error)
        error->clear();
    return true;
}

void reset() {
    gWeights = MovePolicyWeights{};
    gLoaded  = false;
    gPath.clear();
}

const std::string& file_path() { return gPath; }

Context make_context(const Eval::NNUE::PolicyTap& tap, uint8_t bucket, NodeType nodeType, Value psqtLike) {
    Context ctx{};
    if (!enabled())
        return ctx;

    ctx.bucket   = bucket;
    ctx.nodeType = static_cast<uint8_t>(nodeType);
    ctx.psqtLike = static_cast<int16_t>(std::clamp(int(psqtLike), -32768, 32767));
    ctx.enabled  = true;

    alignas(32) int16_t tapVector[2 * Eval::NNUE::L2Big + 1]{};
    for (int i = 0; i < tap.actCount; ++i)
        tapVector[i] = tap.acts[i];
    tapVector[tap.actCount] = tap.fwd;

    for (int k = 0; k < 8; ++k)
    {
        const int sum = dot_i16(gWeights.proj[k], tapVector, 2 * Eval::NNUE::L2Big + 1);
        ctx.h[k] = static_cast<int16_t>(std::clamp(sum / 256, -32768, 32767));
    }

    return ctx;
}

int quiet_bonus(const Context& ctx, const Position& pos, Move move) {
    if (!ctx.enabled || !enabled())
        return 0;

    const int from  = int(move.from_sq());
    const int to    = int(move.to_sq());
    const int piece = int(pos.moved_piece(move));
    const int promo = move.type_of() == PROMOTION ? int(move.promotion_type()) : 0;
    const int flags = int(move.type_of());

    std::array<int16_t, 8> moveVec{};
    for (int i = 0; i < 8; ++i)
        moveVec[i] = static_cast<int16_t>(gWeights.from[from][i] + gWeights.to[to][i]
                                          + gWeights.piece[piece][i] + gWeights.promo[promo][i]
                                          + gWeights.flags[flags][i]);

    int bonus = dot_i16(ctx.h, moveVec.data(), 8);
    bonus += gWeights.bucketBias[ctx.bucket];
    bonus += gWeights.nodeBias[ctx.nodeType];
    bonus += ctx.psqtLike / 8;
    return bonus / 8;
}

std::optional<std::string> load_from_option(const std::string& path) {
    std::string error;
    if (!load(path, &error))
        return error;

    if (path.empty())
        return std::string("Policy head sidecar cleared");

    return std::string("Loaded policy head sidecar: ") + path;
}

}  // namespace Stockfish::Policy
