// Latent MoE: the session-level D/U device pair of each device, stored by the engine's LatentTab and pulled by the
// prefill and verify arms inside their own init (no per-object external wiring to miss).  A non-latent pack
// never stores a pair, so both accessors stay null.
#pragma once
#include <cstdint>

namespace strata {
namespace core {

// The pair is device memory, so there is one per device: a kernel cannot read another device's buffer (HIP answers
// "an illegal memory access was encountered").  A layer split's later stage pulls the pair of its own device, or
// finds none and uploads its own copy.  `device` is the CUDA / HIP device number.
struct LatentSharedPair {
    uint8_t *d = nullptr, *u = nullptr;
};
inline LatentSharedPair& latent_shared(int device) {
    static LatentSharedPair pairs[32];
    return pairs[device & 31];
}
inline uint8_t*& latent_shared_d(int device) { return latent_shared(device).d; }
inline uint8_t*& latent_shared_u(int device) { return latent_shared(device).u; }

inline const uint8_t*& latent_xl_current() { static const uint8_t* p = nullptr; return p; }
inline const uint32_t*& latent_xlseq_current() { static const uint32_t* p = nullptr; return p; }
inline const uint8_t*& latent_act0_current() { static const uint8_t* p = nullptr; return p; }

}  // namespace core
}  // namespace strata
