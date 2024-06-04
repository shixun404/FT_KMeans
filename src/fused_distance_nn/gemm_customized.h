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
  using ThreadblockShape = cutlass::gemm::GemmShape<512, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<256, 16, 16>;
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
//this is for codegen double
template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_tester {
  using ThreadblockShape = cutlass::gemm::GemmShape<512, 32, 32>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
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

//this is for baseline double
template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_0 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_3 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_4 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_5 {
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 256, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_6 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 256, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_7 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 256, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_8 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 256, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_9 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_10 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_11 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_12 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_13 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_14 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_15 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_16 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_17 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_18 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_19 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_20 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_21 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_22 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_23 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_24 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_25 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_26 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_27 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_28 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_29 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_30 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_31 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_32 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_33 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_34 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_35 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_36 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 8>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_37 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_38 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_39 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 8>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_40 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_41 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_42 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 8>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_43 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 8>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_44 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_45 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_46 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 8>;
  using WarpShape = cutlass::gemm::GemmShape<256, 16, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_47 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<256, 16, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_48 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<256, 16, 8>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_49 {
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_50 {
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_51 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_52 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_53 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_54 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_55 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_56 {
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 256, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_57 {
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 512, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 256, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_58 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 256, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_59 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 512, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 256, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_60 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 256, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_61 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 512, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 256, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_62 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 256, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_63 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_64 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_65 {
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_66 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_67 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_68 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_69 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_70 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_71 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_72 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_73 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_74 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 512, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_75 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_76 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_77 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 512, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_78 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_79 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_80 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 128, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_81 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_82 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_83 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_84 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_85 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_86 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_87 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_88 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_89 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_90 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_91 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_92 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_93 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 512, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_94 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_95 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_96 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_97 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_98 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_99 {
  using ThreadblockShape = cutlass::gemm::GemmShape<512, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_100 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_101 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_102 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_103 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_104 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_105 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_106 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_107 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_108 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_109 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_110 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_111 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_112 {
  using ThreadblockShape = cutlass::gemm::GemmShape<512, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_113 {
  using ThreadblockShape = cutlass::gemm::GemmShape<512, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 32, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_114 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<256, 8, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_115 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<256, 8, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_116 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<256, 16, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_117 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<256, 16, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_118 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<256, 16, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_119 {
  using ThreadblockShape = cutlass::gemm::GemmShape<512, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<256, 16, 16>;
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

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_120 {
  using ThreadblockShape = cutlass::gemm::GemmShape<512, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<256, 16, 16>;
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

//double start of injection

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_1 {
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_2 {
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_3 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_4 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_5 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_6 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_7 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_8 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_9 {
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_10 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_11 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_12 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_13 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_14 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_15 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_16 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_17 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_18 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_19 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_20 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_21 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_22 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_23 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_24 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_25 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_26 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_27 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_28 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_29 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_30 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_31 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_32 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_33 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_34 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_35 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_36 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_37 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_38 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_39 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_40 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 256, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_41 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_42 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_43 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_44 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_45 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_46 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_47 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_48 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_49 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_50 {
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_51 {
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_52 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_53 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_54 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_55 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_56 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_57 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_58 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 128, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_59 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_60 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_61 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_62 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_63 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_64 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_65 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_66 {
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_67 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_68 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_69 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_70 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_71 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_72 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_73 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_74 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_75 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_76 {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_77 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_78 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_79 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 32>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};

template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_double_80 {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<128, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using LayoutOutput = cutlass::layout::RowMajor;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type NormXLayout;
  typedef typename std::conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;
  typedef typename std::conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;
  using GemmBase = typename DefaultGemmUniversal<double,LayoutA_,cutlass::ComplexTransform::kNone,1,ElementB_,LayoutB_,cutlass::ComplexTransform::kNone,1,ElementC_,LayoutOutput,ElementAccumulator,OperatorClass,ArchTag,ThreadblockShape,WarpShape,InstructionShape,EpilogueOutputOp,ThreadblockSwizzle,Stages,Operator>::GemmKernel;
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<typename GemmBase::Epilogue::Shape,typename GemmBase::Epilogue::WarpMmaOperator,GemmBase::Epilogue::kPartitionsK,ElementAccumulator,typename EpilogueOutputOp::ElementT,ElementAccumulator,EpilogueOutputOp,NormXLayout,GemmBase::Epilogue::kElementsPerAccess>::Epilogue;
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,Epilogue,ThreadblockSwizzle,GroupScheduleMode::kDeviceOnly>;
};
//double end of injection

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass