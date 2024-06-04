#pragma once

#include <cutlass/cutlass.h>

#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>

#include "epilogue.cuh"
#include "persistent_gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {
//this is for code gen
template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_tester {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAddFastF32;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<ElementA_,LayoutA_,cutlass::ComplexTransform::kNone,kAlignmentA,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,kAlignmentB,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

// this is for baseline
template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_0 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAddFastF32;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<ElementA_,LayoutA_,cutlass::ComplexTransform::kNone,kAlignmentA,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,kAlignmentB,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

//start of injection

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_1 {
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAddFastF32;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<ElementA_,LayoutA_,cutlass::ComplexTransform::kNone,kAlignmentA,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,kAlignmentB,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_2 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAddFastF32;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<ElementA_,LayoutA_,cutlass::ComplexTransform::kNone,kAlignmentA,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,kAlignmentB,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};
//end of injection

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass