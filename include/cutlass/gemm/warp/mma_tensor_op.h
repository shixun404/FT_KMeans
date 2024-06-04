/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Templates implementing warp-level matrix multiply-accumulate operations targeting
      Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/platform/platform.h"

#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h" 
#include "cutlass/arch/mma_sm80.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename T, typename S, int N, FloatRoundStyle Round>
struct ConvertAndPack {

  using Converter = NumericArrayConverter<T, S, N, Round>;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<S, N> const &source) {
    Converter converter;

    return converter(source);
  }
};

template <typename T, int N, FloatRoundStyle Round>
struct ConvertAndPack<T, T, N, Round> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &source) {
		return source;
  }
};

template <int N, FloatRoundStyle Round>
struct ConvertAndPack<bfloat16_t, float, N, Round> {

  using Converter = NumericArrayConverter<bfloat16_t, float, N, Round>;

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(Array<float, N> const &source) {
    Converter converter;

    Array<float, N> tmp;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int idx = (((i << 1) & 2) | ((i >> 1) & 1) | (i & 0xfffffffc));
      tmp[i] = source[idx];
    }

    return converter(tmp);
  }
};

template <int N, FloatRoundStyle Round>
struct ConvertAndPack<half_t, float, N, Round> {

  using Converter = NumericArrayConverter<half_t, float, N, Round>;

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<float, N> const &source) {
    Converter converter;

    Array<float, N> tmp;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int idx = (((i << 1) & 2) | ((i >> 1) & 1) | (i & 0xfffffffc));
      tmp[i] = source[idx];
    }

    return converter(tmp);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Number of partitions along K dimension
  int PartitionsK_ = 1,
  /// Store the accumulators in row major or column major.  Row major is used
  /// when output layout is interleaved.
  bool AccumulatorsInRowMajor = false,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaTensorOp {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Indicates math operator 
  using MathOperator = typename ArchMmaOperator::Operator;

  /// Architecture tag from underlying instruction
  using ArchTag = typename ArchMmaOperator::ArchTag;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Shape of underlying instruction
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = ComplexTransform::kNone;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

public:

  /// Iterates over the A operand in memory
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
     MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
     MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>,
     Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA =
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;

  /// Iterates over the B operand in memory
  using IteratorB = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
      MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
      Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed B tile
  using TransformedFragmentB =
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

  /// Iterates over the C operand in memory
  using IteratorC = MmaTensorOpAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
     typename ArchMmaOperator::Shape, typename Policy::OpDelta>;

  /// Storage for C tile
  using FragmentC = typename IteratorC::Fragment;

  /// Number of mma operations performed
  using MmaIterations = MatrixShape<
    (Shape::kM + ArchMmaOperator::Shape::kM - 1) / ArchMmaOperator::Shape::kM,
    (Shape::kN + ArchMmaOperator::Shape::kN - 1) / ArchMmaOperator::Shape::kN
  >;

public:

  /// Underlying matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaTensorOp() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    TransformedFragmentA const &A, 
    TransformedFragmentB const &B, 
    FragmentC const &C,
    Array<ElementC, ArchMmaOperator::Shape::kMN / 32 * 3> &CC
  ) const {

    using MmaOperandA = typename ArchMmaOperator::FragmentA;
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    D = C;
    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);

    // Array<tfloat32_t, 4> A_;
    // Array<tfloat32_t, 2> B_;
    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
      // Serpentine visitation order maximizing reuse of Rb
      // The visitation order is like
      //      _   
      //   | | | |
      //   | | | |
      //   |_| |_|
      //
      // Down Up Down Up

      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < MmaIterations::kRow; ++m) {

          int m_serpentine = ((n % 2) ? (MmaIterations::kRow - 1 - m) : m);
          // if(threadIdx.x == 0 && blockIdx.x == 0) printf("mma_tensor_op  arch<80  m=%d n=%d\n", m, n);
          if (AccumulatorsInRowMajor) {  // matrix B is reordered
            mma(
              ptr_D[n + m_serpentine * MmaIterations::kColumn],
              ptr_A[m_serpentine],
              ptr_B[n],
              ptr_D[n + m_serpentine * MmaIterations::kColumn]);
          } else {
            mma(
              ptr_D[m_serpentine + n * MmaIterations::kRow],
              ptr_A[m_serpentine],
              ptr_B[n],
              ptr_D[m_serpentine + n * MmaIterations::kRow]);
          }
        }
      }
    #elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      // Serpentine visitation order maximizing reuse of Ra
      // The visitation order is like
      //   _________
      //   _________|
      //  |_________
      //  __________|
      //
      // Right Left Right Left 
      
      Array<typename ArchMmaOperator::ElementB, ArchMmaOperator::Shape::kKN / 32> FragmentI;
      Array<typename ArchMmaOperator::ElementA, ArchMmaOperator::Shape::kMK / 32> tmp;
      Array<typename ArchMmaOperator::ElementC, ArchMmaOperator::Shape::kMN / 32> FragmentA_checksum;
      Array<typename ArchMmaOperator::ElementC, ArchMmaOperator::Shape::kMN / 32> FragmentB_checksum;
      FragmentI.clear();
      tmp.clear();
      FragmentA_checksum.clear();
      FragmentB_checksum.clear();


      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaIterations::kRow; ++m) {

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaIterations::kColumn; ++n) {

          int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);

          if (AccumulatorsInRowMajor) {  // matrix B is reordered
            mma(
              ptr_D[n_serpentine + m * MmaIterations::kColumn],
              ptr_A[m],
              ptr_B[n_serpentine],
              ptr_D[n_serpentine + m * MmaIterations::kColumn]);
            // if(threadIdx.x == 0 && blockIdx.x == 0) printf("mma_tensor_op  AccumulatorsInRowMajor  m=%d n=%d\n", m, n);
          } else {
            mma(ptr_D[m + n_serpentine * MmaIterations::kRow],
                ptr_A[m],
                ptr_B[n_serpentine],
                ptr_D[m + n_serpentine * MmaIterations::kRow]);
                // if(threadIdx.x == 0 && blockIdx.x == 0) printf("mma_tensor_op  not AccumulatorsInRowMajor  m=%d n=%d\n", m, n);
          }
        }
      }
      
      
      FragmentA_checksum[0] = ptr_A[0][0]; 
      FragmentA_checksum[1] = ptr_A[0][0] * 1; 
      CUTLASS_PRAGMA_UNROLL
      for (int m = 1; m < MmaIterations::kRow; ++m) {
        int n = 0;
         int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);
        // tfloat32_t a((int)(((threadIdx.x % 4) == (threadIdx.x / 8)) * (1 + ((threadIdx.x / 4) % 2) * m)));
        // tfloat32_t a((int)0);
        // FragmentI[0] = ((((threadIdx.x % 4) == (threadIdx.x / 8)) * (1 + ((threadIdx.x / 4) % 2) * m)));
        FragmentA_checksum[0] += ptr_A[m][0]; 
        // FragmentA_checksum[1] += ptr_A[m][0] * m; 
        // FragmentA_checksum[1] += ptr_A[m][0] * m; 
        // FragmentA_checksum[3] += ptr_A[m][1] * m; 
          mma(
            FragmentA_checksum,
            // ptr_D[m + n_serpentine * MmaIterations::kRow],
              ptr_A[m],
              FragmentI,
              FragmentA_checksum
              // ptr_D[m + n_serpentine * MmaIterations::kRow]
              );
              // if(threadIdx.x == 0 && blockIdx.x == 0) printf("mma_tensor_op  not AccumulatorsInRowMajor  m=%d n=%d\n", m, n);
      }

      FragmentB_checksum[0] = ptr_B[0][0]; 
      // FragmentB_checksum[1] = ptr_B[0][0] * 1; 
      CUTLASS_PRAGMA_UNROLL
      for (int n = 1; n < MmaIterations::kColumn; ++n) {

        int m = 0;
        
        // B_[1] += ptr_B[n][1]; 
        // B_[0] += B_[1]; 
        // B_[1] += B_[0]; 
        // ptr_A[]
        int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);
        
        // double a((int)(((threadIdx.x % 4) == (threadIdx.x / 8)) * (1 + ((threadIdx.x / 4) % 2) * n_serpentine)));
        // tfloat32_t a((int)0);
        // FragmentI[0] = (((threadIdx.x % 4) == (threadIdx.x / 8)) * (1 + ((threadIdx.x / 4) % 2) * n_serpentine));
          // tmp[0] = ptr_B[n_serpentine][0];
      
        FragmentB_checksum[0] += ptr_B[n_serpentine][0]; 
        // FragmentB_checksum[1] += ptr_B[n_serpentine][0] * n_serpentine; 
        // if(threadIdx.x == 0 && blockIdx.x == 0) printf("FT==1\n");
        //  FragmentB_checksum[0] +=  FragmentB_checksum[1];
          mma(
            FragmentB_checksum,
            //  ptr_D[m + n_serpentine * MmaIterations::kRow],
              ptr_B[n_serpentine],
              FragmentI,
              FragmentB_checksum
              //  ptr_D[m + n_serpentine * MmaIterations::kRow]
              );
          // if(threadIdx.x == 0 && blockIdx.x == 0) printf("FT==2\n");
              // if(threadIdx.x == 0 && blockIdx.x == 0) printf("mma_tensor_op  not AccumulatorsInRowMajor  m=%d n=%d\n", m, n);
      }

      {
        int m = 0, n_serpentine = 0;
      double const *A = reinterpret_cast<double const *> (&(FragmentA_checksum[0]));
      double const *B = reinterpret_cast<double const *> (&(FragmentB_checksum[0]));

      asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=d"(CC[0]), "=d"(CC[1])
      : "d"(A[0]), "d"(B[0]), "d"(CC[0]), "d"(CC[1]));

      // asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      // : "=d"(CC[2]), "=d"(CC[3])
      // : "d"(A[1]), "d"(B[0]), "d"(CC[2]), "d"(CC[3]));

      // asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      // : "=d"(CC[4]), "=d"(CC[5])
      // : "d"(A[0]), "d"(B[1]), "d"(CC[4]), "d"(CC[5]));
      
      }


    #else
      assert(0);
    #endif
  }

  CUTLASS_DEVICE
  void error_detector_operator(
    FragmentC &D, 
    Array<ElementC, ArchMmaOperator::Shape::kMN / 32 * 3> &CC
  ) const {

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

      using MmaOperandA = typename ArchMmaOperator::FragmentA;
      using MmaOperandB = typename ArchMmaOperator::FragmentB;
      using MmaOperandC = typename ArchMmaOperator::FragmentC;
      
      MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

      {
        int m = 0, n_serpentine = 0;
      float delta_error = CC[0];
      float D_sum = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaIterations::kRow; ++m) {

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaIterations::kColumn; ++n) {
          
          // This allows to reuse of Rb when at serpentine turns
          int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);
          
           CC[0] -= ptr_D[m + n_serpentine * MmaIterations::kRow][0];
           D_sum += ptr_D[m + n_serpentine * MmaIterations::kRow][0];
          //  CC[0] -= ptr_D[m + n_serpentine * MmaIterations::kRow][1];
          //  CC[0] -= ptr_D[m + n_serpentine * MmaIterations::kRow][2];
          //  CC[0] -= ptr_D[m + n_serpentine * MmaIterations::kRow][3];
        } // end n loop
      } // end m loop     
      // if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0  && blockIdx.x == 0) printf("CC[0]_initial  %f CC[0]=%f D_sum=%f\n", delta_error, CC[0], D_sum);
      ptr_D[0][0] += (CC[0] + CC[2] + CC[4]) * 0.00001;
      }

    #else
      assert(0);
    #endif
  }


  /// Transform the mma operands to the required types
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {

    //
    // Define conversions from source type to instruction type
    //
    FloatRoundStyle const kRoundA =
        PreferredRoundingMode<typename ArchMmaOperator::ElementA,
                              ElementA>::kRound;
    FloatRoundStyle const kRoundB =
        PreferredRoundingMode<typename ArchMmaOperator::ElementB,
                              ElementB>::kRound;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
      detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            FragmentA::kElements, kRoundA>
          convert_A;
      NumericArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                            FragmentB::kElements / 2, kRoundB>
          convert_B;
      Array<ElementB, FragmentB::kElements / 2> const *ptr_B =
          reinterpret_cast<Array<ElementB, FragmentB::kElements / 2> const *>(&B);
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements / 2> *
          ptr_dst_B = reinterpret_cast<Array<typename ArchMmaOperator::ElementB,
                                             FragmentB::kElements / 2> *>(&dst_B);
  
      dst_A = convert_A(A);
  
      ptr_dst_B[0] = convert_B(ptr_B[0]);
      ptr_dst_B[1] = convert_B(ptr_B[1]);

    #elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            FragmentA::kElements / 2, kRoundA>
          convert_A;
      NumericArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                            FragmentB::kElements, kRoundB>
          convert_B;
      Array<ElementA, FragmentA::kElements / 2> const *ptr_A =
          reinterpret_cast<Array<ElementA, FragmentA::kElements / 2> const *>(&A);
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements / 2> *
          ptr_dst_A = reinterpret_cast<Array<typename ArchMmaOperator::ElementA,
                                             FragmentA::kElements / 2> *>(&dst_A);
  
      dst_B = convert_B(B);
  
      ptr_dst_A[0] = convert_A(ptr_A[0]);
      ptr_dst_A[1] = convert_A(ptr_A[1]);
    #else
      assert(0);
    #endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/warp/mma_tensor_op_fast_f32.h"

/////////////////////////////////////////////////////////////////////////////////////////////////