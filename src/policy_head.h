#ifndef POLICY_HEAD_H_INCLUDED
#define POLICY_HEAD_H_INCLUDED

#include <cstdint>
#include <optional>
#include <string>

#include "nnue/nnue_architecture.h"
#include "types.h"

namespace Stockfish {

class Position;

namespace Policy {

enum class NodeType : uint8_t { PV, Cut, All };

struct Context {
    int16_t h[8]{};
    int16_t psqtLike = 0;
    uint8_t bucket   = 0;
    uint8_t nodeType = 0;
    bool    enabled  = false;
};

struct StaticEvalInfo {
    Value   eval      = VALUE_ZERO;
    Context policy{};
    bool    hasPolicy = false;
};

void                       set_enabled(bool enabled);
bool                       enabled();
bool                       load(const std::string& path, std::string* error = nullptr);
void                       reset();
const std::string&         file_path();
Context                    make_context(const Eval::NNUE::PolicyTap&, uint8_t, NodeType, Value);
int                        quiet_bonus(const Context&, const Position&, Move);
std::optional<std::string> load_from_option(const std::string& path);

}  // namespace Policy

}  // namespace Stockfish

#endif  // #ifndef POLICY_HEAD_H_INCLUDED
