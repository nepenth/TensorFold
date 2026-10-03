// One projection selected from CUDA pointer tables. The owners stay in RoutedTable;
// these non-owning views use the existing lane CUDA implementation and K slices.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>

void lane_cuda(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double,
               at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, bool);

namespace {

void table_in(const at::Tensor& t, at::ScalarType type, const at::Tensor& codes, int64_t experts) {
    TORCH_CHECK(t.is_cuda() && t.device() == codes.device() && t.scalar_type() == type && t.is_contiguous() &&
                t.dim() == 2 && t.size(0) == experts && t.size(1) == 3,
                "tables: contiguous CUDA (experts, 3), words_ptr/bs_ptr int64, n/k int32, alpha fp32");
}

bool intersects(uintptr_t a, size_t na, uintptr_t b, size_t nb) {
    return na && nb && (a <= b ? b - a < na : a - b < nb);
}

void refuse_overlap(const at::Tensor& out, uintptr_t ptr, size_t bytes) {
    // Check the whole destination storage, including offset views of that storage.
    TORCH_CHECK(!intersects(reinterpret_cast<uintptr_t>(out.storage().data_ptr().get()), out.storage().nbytes(),
                           ptr, bytes), "destination must not overlap input, tables, or projection storage");
}

void dispatch(const at::Tensor& codes, const at::Tensor& scales, const at::Tensor& words_ptr,
              const at::Tensor& bs_ptr, const at::Tensor& ns, const at::Tensor& ks, const at::Tensor& alphas,
              int64_t expert, int64_t projection, at::Tensor out, int64_t sk, int64_t tile) {
    TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == at::kByte && codes.is_contiguous() &&
                codes.dim() == 2 && codes.size(0) > 0, "codes: contiguous CUDA uint8 (positive rows, K/2)");
    const int64_t experts = words_ptr.dim() == 2 ? words_ptr.size(0) : 0;
    table_in(words_ptr, at::kLong, codes, experts);
    table_in(bs_ptr, at::kLong, codes, experts);
    table_in(ns, at::kInt, codes, experts);
    table_in(ks, at::kInt, codes, experts);
    table_in(alphas, at::kFloat, codes, experts);
    TORCH_CHECK(expert >= 0 && expert < experts, "expert id is out of range");
    TORCH_CHECK(projection >= 0 && projection < 3, "projection column must be gate=0, up=1, or down=2");
    c10::cuda::CUDAGuard guard(codes.device());

    // Task 10A is eager: item() reads the selected device slot on the current stream.
    // NEVER narrow an address through int32, even though n and k are int32 slots.
    static_assert(sizeof(uintptr_t) == sizeof(int64_t), "dispatch requires 64-bit addresses");
    const auto wp = static_cast<uintptr_t>(words_ptr.select(0, expert).select(0, projection).item<int64_t>());
    const auto sp = static_cast<uintptr_t>(bs_ptr.select(0, expert).select(0, projection).item<int64_t>());
    const int64_t n = ns.select(0, expert).select(0, projection).item<int32_t>();
    const int64_t k = ks.select(0, expert).select(0, projection).item<int32_t>();
    const float alpha = alphas.select(0, expert).select(0, projection).item<float>();
    TORCH_CHECK(wp && sp && wp % 16 == 0 && sp % 16 == 0, "projection addresses must be nonzero and aligned");
    TORCH_CHECK(n > 0 && k > 0 && k % 64 == 0 && sk > 0 && sk <= 8 && (k / 64) % sk == 0,
                "n/K must be positive, K in whole steps of 64 split evenly into 1..8 slices");
    TORCH_CHECK(std::isfinite(alpha), "alpha must be finite");
    const int64_t m = codes.size(0), npad = (n + 63) / 64 * 64;
    TORCH_CHECK(codes.size(1) == k / 2, "codes width must match the selected K/2");
    TORCH_CHECK(scales.is_cuda() && scales.device() == codes.device() && scales.scalar_type() == at::kByte &&
                scales.is_contiguous() && scales.dim() == 3 && scales.size(0) == k / 64 &&
                scales.size(1) >= m && scales.size(1) % 64 == 0 && scales.size(2) == 4,
                "scales: contiguous CUDA uint8 (K/64, mpad, 4), mpad a multiple of 64 holding every row");
    TORCH_CHECK(out.is_cuda() && out.device() == codes.device() && out.dim() == 2 &&
                out.size(0) == m && out.size(1) == n, "destination must have shape (rows, n) on the input device");
    TORCH_CHECK(out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16,
                "destination must be fp32 or bf16");
    TORCH_CHECK(out.is_contiguous(), "destination must be contiguous and have no internal overlap");
    for (const auto* source : {&codes, &scales, &words_ptr, &bs_ptr, &ns, &ks, &alphas})
        refuse_overlap(out, reinterpret_cast<uintptr_t>(source->storage().data_ptr().get()),
                       source->storage().nbytes());
    refuse_overlap(out, wp, npad * k / 2);
    refuse_overlap(out, sp, (k / 64) * npad * 4);

    // The overload without a deleter borrows storage; it MUST NOT free either address.
    const auto w = at::from_blob(reinterpret_cast<void*>(wp), {npad / 64, k / 64, 8, 32, 2},
                                 codes.options().dtype(at::kInt));
    const auto ws = at::from_blob(reinterpret_cast<void*>(sp), {npad / 64, k / 64, 64, 4}, codes.options());
    lane_cuda(0, codes, scales, w, ws, alpha, out, at::Tensor(), n, k, sk, scales.size(1), tile,
              out.scalar_type() == at::kFloat);
}

}  // namespace

void bind_dispatch(pybind11::module_& m) {
    m.def("dispatch", &dispatch);
}
