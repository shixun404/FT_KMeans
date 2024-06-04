# part1: use main.cu to generate every possible parameters
#     1. search for all parameters and put together into a list
#     2. try all possible parameters, and choose those successful ones

# part2: transfer the parameters to src/test_kmeans_tuner.cu (change the maximum size of possible parameters)
# and src/fused_distance_nn/cutlass_base_customized.cuh and (add an long if to choose a kernel)
# src/fused_distance_nn/gemm_customized.cuh (put all possible kernel parameters here!)
import subprocess
def modify_parameter(parameter_value):
    cpp_file = "../src/fused_distance_nn/gemm_customized.h"
    with open(cpp_file, "r") as file:
        lines = file.readlines()
        
    for i, line in enumerate(lines):
        if "GEMM_float_tester" in line:
            new_line = f"  using ThreadblockShape = cutlass::gemm::GemmShape<{parameter_value[0]}>;\n"
            lines[i + 1] = new_line
            new_line = f"  using WarpShape = cutlass::gemm::GemmShape<{parameter_value[1]}>;\n"
            lines[i + 2] = new_line
            new_line = f"  using InstructionShape = cutlass::gemm::GemmShape<{parameter_value[2]}>;\n"
            lines[i + 3] = new_line
            break
            
    with open(cpp_file, "w") as file:
        file.writelines(lines)
    
def generate_parameter(type):
    if type == "float":
        params = []
        para = ["","",""]
        para_op = [16, 8, 4]
        para_warp = [16, 8, 4]
        para_block = [16, 8, 4]
        for k in range(3):
            para_block[2] = para_warp[2]
            for i in range(5):
                para_warp[1] = para_op[0] * para_op[1] * 16 / para_warp[0] 
                para_block[0] = para_warp[0]
                for j in range(4):
                    para_block[1] = para_warp[1]
                    div = int(para_warp[0] * 8 / para_block[0])
                    for k in range(div):
                        if (para_block[0] <= 1024 and para_block[1] <= 1024 and para_warp[2] >= 8):
                            para[0] = f"{para_block[0]}, {int(para_block[1])}, {para_block[2]}"
                            para[1] = f"{para_warp[0]}, {int(para_warp[1])}, {para_warp[2]}"
                            para[2] = f"{para_op[0]}, {para_op[1]}, {para_op[2]}"
                            params.append(para.copy())
                        para_block[1] = para_block[1] * 2
                    para_block[0] = para_block[0] * 2
                
                para_warp[1] = para_op[0] * para_op[1] * 32 / para_warp[0] 
                para_block[0] = para_warp[0]
                for j in range(5):
                    para_block[1] = para_warp[1]
                    div = int(para_warp[0] * 8 / para_block[0])
                    for k in range(div):
                        if (para_block[0] <= 512 and para_block[1] <= 512 and para_warp[2] >= 8):
                            para[0] = f"{para_block[0]}, {int(para_block[1])}, {para_block[2]}"
                            para[1] = f"{para_warp[0]}, {int(para_warp[1])}, {para_warp[2]}"
                            para[2] = f"{para_op[0]}, {para_op[1]}, {para_op[2]}"
                            params.append(para.copy())
                        para_block[1] = para_block[1] * 2
                    para_block[0] = para_block[0] * 2
                
                para_warp[0] = para_warp[0] * 2
            para_warp[0] = para_op[0]
            para_warp[2] = para_warp[2] * 2
        return params

def check_success():
    output_file = "../output.txt"
    with open(output_file, "r") as file:
        lines = file.readlines()
    #print(lines)
    #print(len(lines))
    if len(lines) == 3:
        return True
    return False

def compile_program():
    subprocess.run("make -C ../ clean", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    subprocess.run("make -C ../ main >& ../output.txt", shell=True, check=False)
    return check_success()

def check_res():
    output_file = "../output1.txt"
    with open(output_file, "r") as file:
        lines = file.readlines()
    #print(lines)
    if len(lines) == 0:
        return True
    return False

def run_program():
    try:
        subprocess.run("../main > ../output1.txt", shell=True, check=False)
        if check_res():
            return True
        return False
    except:
        return False
    
def create_function(item, num):
    myfunc = f"""
template <typename ElementA_, int kAlignmentA, typename ElementB_, int kAlignmentB, typename ElementC_, typename ElementAccumulator, typename EpilogueOutputOp, int Stages, bool isRowMajor> struct GEMM_float_{num} {{
  using ThreadblockShape = cutlass::gemm::GemmShape<{item[0]}>;
  using WarpShape = cutlass::gemm::GemmShape<{item[1]}>;
  using InstructionShape = cutlass::gemm::GemmShape<{item[2]}>;
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
}};
"""
    return myfunc

def create_funcs(num):
    myfuncs = f"""
template <typename DataT,typename AccT,typename OutT,typename IdxT,int VecLen,typename CGReduceOpT,typename DistanceFn,typename ReduceOpT,typename KVPReduceOpT>
void cutlassFusedDistanceNN_{num}(const DataT* x,const DataT* y,const DataT* xn,const DataT* yn,IdxT m,IdxT n,IdxT k,IdxT lda,IdxT ldb,IdxT ldd,OutT* dOutput,int* mutexes,CGReduceOpT cg_reduce_op,DistanceFn dist_op,ReduceOpT redOp,KVPReduceOpT pairRedOp,cudaStream_t stream){{
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<DataT,AccT,DataT,AccT,OutT,1,DistanceFn,CGReduceOpT,ReduceOpT,KVPReduceOpT>;
  constexpr int batch_count = 1;
  typename EpilogueOutputOp::Params epilog_op_param(dist_op, cg_reduce_op, redOp, pairRedOp, mutexes); 
  constexpr int NumStages = 3; constexpr int Alignment = VecLen;
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k); constexpr bool isRowMajor = true;
  using fusedDistanceNNKernel = typename cutlass::gemm::kernel::GEMM_float_{num}<DataT,Alignment,DataT,Alignment,AccT,AccT,EpilogueOutputOp,NumStages,isRowMajor>::GemmKernel;
  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;
  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = 108;
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;
  typename fusedDistanceNN::Arguments arguments{{problem_size,batch_count,thread_blocks,epilog_op_param,x,y,xn,(DataT*)yn,dOutput,(int64_t)lda,(int64_t)ldb,(int64_t)1,(int64_t)ldd}};
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  DataT* workspace;
  cudaMalloc((void**)&workspace, workspace_size);
  fusedDistanceNN fusedDistanceNN_op;
  fusedDistanceNN_op.can_implement(arguments);
  fusedDistanceNN_op.initialize(arguments, workspace, stream);
  fusedDistanceNN_op.run(stream);
}}
"""
    return myfuncs

def create_ifs(num):
    myif = f"""
    if (num == {num}) cutlassFusedDistanceNN_{num}<DataT,DataT,OutT,IdxT,16 / sizeof(DataT),kvp_cg_min_reduce_op_,L2Op,MinAndDistanceReduceOp,KVPMinReduce>(dx,dy,dxn,dyn,m,n,k,lda,ldb,ldd,dmin,dworkspace,cg_reduce_op,L2_dist_op,redOp,pairRedOp,stream);
"""
    return myif

def write_kernel(func):
    Fi = "../src/fused_distance_nn/gemm_customized.h"
    with open(Fi, "r") as file:
        lines = file.readlines()
    x = 0
    y = 0
    for i, line in enumerate(lines):
        if "//start of injection" in line:
            x = i
        if "//end of injection" in line:
            y = i
            break
    
    lines_new = []
    for i in range(x+1):
        lines_new.append(lines[i])
    
    lines_new.append(func)
    
    for i in range(len(lines) - y):
        lines_new.append(lines[y+i])

    with open(Fi, "w") as file:
        file.writelines(lines_new)
        
def write_funcs(funcs):
    Fi = "../src/fused_distance_nn/cutlass_base_customized.cuh"
    with open(Fi, "r") as file:
        lines = file.readlines()
    x = 0
    y = 0
    for i, line in enumerate(lines):
        if "//start of injection" in line:
            x = i
        if "//end of injection" in line:
            y = i
            break
    
    lines_new = []
    for i in range(x+1):
        lines_new.append(lines[i])
    
    lines_new.append(funcs)
    
    for i in range(len(lines) - y):
        lines_new.append(lines[y+i])

    with open(Fi, "w") as file:
        file.writelines(lines_new)
        
def write_ifs(ifs):
    Fi = "../src/test_kmeans_tuner.cu"
    with open(Fi, "r") as file:
        lines = file.readlines()
    x = 0
    y = 0
    for i, line in enumerate(lines):
        if "//start of injection" in line:
            x = i
        if "//end of injection" in line:
            y = i
            break
    
    lines_new = []
    for i in range(x+1):
        lines_new.append(lines[i])
    
    lines_new.append(ifs)
    
    for i in range(len(lines) - y):
        lines_new.append(lines[y+i])

    with open(Fi, "w") as file:
        file.writelines(lines_new)

def write_range(ran):
    Fi = "../src/test_kmeans_tuner.cu"
    with open(Fi, "r") as file:
        lines = file.readlines()
    x = 0
    y = 0
    for i, line in enumerate(lines):
        if "IdxT num" in line:
            lines[i] = f"    IdxT num = {ran};\n"

    with open(Fi, "w") as file:
        file.writelines(lines)

outer = generate_parameter("float")
pos_pra = []
for i in range(len(outer)):
    print("Now running:", i)
    print(outer[i][0], outer[i][1], outer[i][2])
    modify_parameter(outer[i].copy())
    if (compile_program() == True and run_program() == True):
        print("success!")
        pos_pra.append(outer[i].copy())

# for i in range(len(pos_pra)):
#     print(pos_pra[i])

with open("output.txt", "w") as file:
    for i in range(len(pos_pra)):
        file.write(str(pos_pra[i][0]) + ' ' + str(pos_pra[i][1])+ ' ' + str(pos_pra[i][2]) + '\n')
# print("======End of parameter generation")
# #===== End of Part 1 =====

# func = ""
# funcs = ""
# ifs = ""
# # always use i+1, cuz i starts from 1!!!
# for i in range(len(pos_pra)):
#     func = func + create_function(pos_pra[i], i + 1)
#     funcs = funcs + create_funcs(i + 1)
#     ifs = ifs + create_ifs(i + 1)

# write_kernel(func)
# write_funcs(funcs)
# write_range(len(pos_pra))
# write_ifs(ifs)