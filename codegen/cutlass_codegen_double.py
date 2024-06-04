import subprocess
def modify_parameter(parameter_value):
    cpp_file = "../src/test_cutlass_double.cu"
    with open(cpp_file, "r") as file:
        lines = file.readlines()
        
    for i, line in enumerate(lines):
        if "//injection here" in line:
            new_line = f"  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<{parameter_value[0]}>;\n"
            lines[i + 1] = new_line
            new_line = f"  using ShapeMMAWarp = cutlass::gemm::GemmShape<{parameter_value[1]}>;\n"
            lines[i + 2] = new_line
            new_line = f"  using ShapeMMAOp = cutlass::gemm::GemmShape<{parameter_value[2]}>;\n"
            lines[i + 3] = new_line
            break
            
    with open(cpp_file, "w") as file:
        file.writelines(lines)
    
def generate_parameter(type):
    if type == "double":
        params = []
        para = ["","",""]
        para_op = [8, 8, 4]
        para_warp = [8, 8, 4]
        para_block = [8, 8, 4]
        for k in range(4):
            para_block[2] = para_warp[2]
            for i in range(5):
                para_warp[1] = para_op[0] * para_op[1] * 16 / para_warp[0] 
                para_block[0] = para_warp[0]
                for j in range(4):
                    para_block[1] = para_warp[1]
                    div = int(para_warp[0] * 8 / para_block[0])
                    for k in range(div):
                        if (para_block[0] <= 1024 and para_block[1] <= 1024 and para_warp[2] >= 16):
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
                        if (para_block[0] <= 512 and para_block[1] <= 512 and para_warp[2] >= 16):
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
    #subprocess.run("make -C ../ clean", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    subprocess.run("make -C ../ test_cutlass_double >& ../output.txt", shell=True, check=False)
    return check_success()

def run_program():
    try:
        subprocess.run("../test_cutlass_double", shell=True, check=False)
        return True
    except:
        return False
    
def create_function(item, num):
    myfunc = f"""
bool CutlassDgemmNN_{num}(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {{
  using ElementAccumulator = DataT;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = DataT;
  using ElementInputB = DataT;
  using ElementOutput = DataT;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<{item[0]}>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<{item[1]}>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<{item[2]}>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementComputeEpilogue>; 
  constexpr int NumStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  Gemm gemm_op;
  Gemm::Arguments arguments({{M, N, K}}, {{A, lda}}, {{B, ldb}}, {{C, ldc}}, {{C, ldc}}, {{alpha, beta}});
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) return 0; return 1;
}}
"""
    return myfunc

def create_header(num):
    myheader = f"bool CutlassDgemmNN_{num}(int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc);\n"
    return myheader

def create_arr(num):
    myif = "    Func Arr[] = {"
    for i in range(num):
        myif = myif + f"CutlassDgemmNN_{i + 1}"
        if i+1 != num:
            myif = myif + ","
    myif = myif + "};\n"
    return myif

def write_func(func):
    Fi = "../src/test_cutlass_tuner_double.cu"
    with open(Fi, "r") as file:
        lines = file.readlines()
    x = 0
    y = 0
    for i, line in enumerate(lines):
        if "//start injection func" in line:
            x = i
        if "//end injection func" in line:
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
        
def write_arr(arr):
    Fi = "../src/test_cutlass_tuner_double.cu"
    with open(Fi, "r") as file:
        lines = file.readlines()
    x = 0
    y = 0
    for i, line in enumerate(lines):
        if "//start injection arr" in line:
            x = i
        if "//end injection arr" in line:
            y = i
            break
    
    lines_new = []
    for i in range(x+1):
        lines_new.append(lines[i])
    
    lines_new.append(arr)
    
    for i in range(len(lines) - y):
        lines_new.append(lines[y+i])

    with open(Fi, "w") as file:
        file.writelines(lines_new)

def write_header(header):
    Fi = "../src/test_cutlass_tuner_double.cu"
    with open(Fi, "r") as file:
        lines = file.readlines()
    x = 0
    y = 0
    for i, line in enumerate(lines):
        if "//start injection header" in line:
            x = i
        if "//end injection header" in line:
            y = i
            break
    
    lines_new = []
    for i in range(x+1):
        lines_new.append(lines[i])
    
    lines_new.append(header)
    
    for i in range(len(lines) - y):
        lines_new.append(lines[y+i])

    with open(Fi, "w") as file:
        file.writelines(lines_new)

outer = generate_parameter("double")
pos_pra = []
for i in range(len(outer)):
    print("Now running:", i)
    print(outer[i][0], outer[i][1], outer[i][2])
    modify_parameter(outer[i].copy())
    if (compile_program() == True and run_program() == True):
        print("success!")
        pos_pra.append(outer[i].copy())

print("======End of parameter generation")
#===== End of Part 1 =====

func = ""
header = ""
arrs = ""
# always use i+1, cuz i starts from 1!!!
for i in range(len(pos_pra)):
    func = func + create_function(pos_pra[i], i + 1)
    header = header + create_header(i + 1)
arrs = create_arr(len(pos_pra))

write_func(func)
write_header(header)
write_arr(arrs)