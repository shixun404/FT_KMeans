/**
 * Reserve 1 digit of precision from each floating-point type
 * for round-off error tolerance.
 * @tparam DataT
 */
template <typename DataT>
__device__ constexpr DataT get_clamp_precision()
{
  switch (sizeof(DataT)) {
    case 2: return 1e-3;
    case 4: return 1e-6;
    case 8: return 1e-15;
    default: return 0;
  }
}

// Epilogue operator for CUTLASS based kernel
template <typename DataT, typename AccT>
struct l2_exp_cutlass_op {
  bool sqrt;

  __host__ __device__ l2_exp_cutlass_op() noexcept : sqrt(false) {}
  __host__ __device__ l2_exp_cutlass_op(bool isSqrt) noexcept : sqrt(isSqrt) {}
  inline __host__ __device__ AccT operator()(DataT aNorm, DataT bNorm, DataT accVal) const noexcept
  {
    AccT outVal = aNorm + bNorm - DataT(2.0) * accVal;

    /**
     * Self-neighboring points should have (aNorm == bNorm) == accVal and the dot product (accVal)
     * can sometimes have round-off errors, which will cause (aNorm == bNorm) ~ accVal instead.
     */
    outVal = outVal * !((outVal * outVal < get_clamp_precision<DataT>()) * (aNorm == bNorm));
    return sqrt ? 0 : outVal;
  }

  __host__ __device__ AccT operator()(DataT aData) const noexcept { return aData; }
};
