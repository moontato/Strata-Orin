#pragma once
namespace strata::core {
class UnifiedKvVmm;
// Construction-time only. Legacy sizing/carving is unchanged when disabled.
inline bool unified_qsa_enabled = false;
inline thread_local UnifiedKvVmm* unified_qsa_owner = nullptr;
class UnifiedKvBinding {
    UnifiedKvVmm* previous_;
public:
    explicit UnifiedKvBinding(UnifiedKvVmm* owner) : previous_(unified_qsa_owner) { unified_qsa_owner = owner; }
    ~UnifiedKvBinding() { unified_qsa_owner = previous_; }
    UnifiedKvBinding(const UnifiedKvBinding&) = delete;
    UnifiedKvBinding& operator=(const UnifiedKvBinding&) = delete;
};
} // namespace strata::core
