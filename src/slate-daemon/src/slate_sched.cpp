#include <algorithm>
#include "slate_sched.h"
#include "client_table.h"

void FairScheduler::update_sched()
{
    //static int old_size = 0;
    unsigned int size = 0;
    cudaDeviceProp props;
    ClientTable& c = ClientTable::getInstance();

    get_props(&props);
    c.lock_table();
    if (SlateClient::getRunningClients())// && c.size() != old_size)
    {
        size = props.multiProcessorCount/SlateClient::getRunningClients();
        if (SlateClient::getRunningClients() == 1 && SlateClient::getHeldClients() == 1)
        {
            c.update_clients(0, 0u, 0u);
            c.update_clients(1, size, 0u);
        }
        else
        {
            c.update_clients(0, 0u, size);
            c.update_clients(1, size, size);
        }
        c.reschedule();
    }
    //old_size = c.size();
    c.unlock_table();
}

std::string FairScheduler::gen_scheduler(std::string& signature, const char* kernel_name, dim3& blocks, dim3& threads, size_t& smem)
{
    cudaDeviceProp props;
    get_props(&props);
    size_t phase_blocks = props.multiProcessorCount * props.maxThreadsPerMultiProcessor / (threads.x * threads.y * threads.z);
    //size_t total_blocks = (blocks.x * blocks.y * blocks.z);
    //phase_blocks = phase_blocks < total_blocks ? phase_blocks : total_blocks;
    size_t start = 1;
    size_t end = signature.size();
    std::list<std::string> args;
    while (start < end + 1)
    {
        size_t comma_pos;
        comma_pos = signature.find(",", start);
        if (comma_pos == std::string::npos) 
        {
               comma_pos = end;
        }
        start = comma_pos + 1;
        size_t word_pos = comma_pos - 1;
        while(isspace(signature[word_pos]) || signature[word_pos] == ']') 
        {
            if (signature[word_pos] == ']')
            {
                while (signature[word_pos]!= '[') word_pos--;
                comma_pos = word_pos - 1;
            }
            word_pos--;
        }
        if (signature[word_pos] == '(') break;
        while(isalnum(signature[word_pos]) || signature[word_pos] == '_') word_pos--;
        //printf("word_pos, l: %d, %d\n", word_pos, comma_pos - word_pos + 1);
        args.push_back(signature.substr(word_pos, comma_pos - word_pos + 1));
        //printf("last arg? %s\n", args.back().c_str());
    //    args.push_back(", ");
    }
    //if (!args.empty()) args.pop_back();
    std::string arg_string;
    if (signature.size() > 0)
    {   
        signature = std::string(", ") + signature;
        arg_string += ",";
    }
    while(!args.empty())
    {
        arg_string += args.front();
        args.pop_front();
    }
    std::string official_kername(kernel_name);
    official_kername.erase( std::remove( official_kername.begin(), official_kername.end(), '<' ), official_kername.end() );
    official_kername.erase( std::remove( official_kername.begin(), official_kername.end(), '>' ), official_kername.end() );
    return  
"//const __device__ uint3 grid = {" + std::to_string(blocks.x) + ", " + std::to_string(blocks.y) + ", " + std::to_string(blocks.z) +"};\n\
const __device__ uint3 grid = {" + std::to_string(phase_blocks) + ", 1, 1};\n\
const __device__ uint3 block = {" + std::to_string(threads.x) + ", " + std::to_string(threads.y) + ", " + std::to_string(threads.z) +"};\n"
 + std::string("extern \"C\" __global__ void ") + std::string(official_kername) + std::string("scheduler(int reps, volatile uint* start_sm, volatile uint* end_sm") + signature +
std::string(") {\n\
    cudaStream_t s;\n\
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);\n\
    retreat = 0;\n\
    //printf(\"sm hi, low: %u, %u\\n\", *start_sm, *end_sm);\n\
    //printf(\"reps: %u, %u\\n\", reps);\n\
    //return;\n\
    //for (int i = 0; i < reps; i++)\n  //{\n\
        slate_idx = 0;\n\
        do\n\
        {\n\
          //  living_blocks = 0;\n\
            ") +
         kernel_name + std::string("<<<grid, block, " + std::to_string(smem) + ", s>>>(*start_sm, *end_sm") + arg_string + std::string(");\n\
         cudaDeviceSynchronize();\n\
         retreat = 0;\n\
        } while (slate_idx < slate_max);\n\
    //}\n\
    cudaStreamDestroy(s);\
}\n");//}");
}

 //printf(\"launch_blocks, old_total, total_finished:  %u, %u, %u\\n\", launch_blocks, old_total, blocks_finished);\n
void FairScheduler::get_params(std::vector<CUdeviceptr>& params)
{
    params.clear();
    params.reserve(2);
    CUdeviceptr sm1;
    CUdeviceptr sm2;
    CUDA_SAFE_CALL(cuMemAlloc(&sm1, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cuMemAlloc(&sm2, sizeof(unsigned int)));
    //printf("sm1, sm2: %llu, %llu\n", sm1, sm2);
    CHECK_CUDA_ERROR();
    params.push_back(sm1);
    params.push_back(sm2);
/*
    CUDA_SAFE_CALL(cuMemcpyHtoD(start_sm, &hosts, sizeof(size_t)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(end_sm, &hosts, sizeof(size_t)));
    CHECK_CUDA_ERROR();
*/
}
/*
    tl.x = (thread_id % (bdx * bdy)) % bdx;\n\
    tl.y = (thread_id % (bdx * bdy * bdz)) / (bdx * bdy);\n\
    tl.z = (thread_id / (bdx * bdy * bdz));\n\
*/


