#pragma once

class FairScheduler;

#include <string>
#include <list>
#include <vector>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "safecuda.h"


class FairScheduler
{
    public:
    
    static void update_sched();
    static std::string gen_scheduler(std::string& signature, const char* kernel_name, dim3& blocks, dim3& threads, size_t& smem);
    static void get_params(std::vector<CUdeviceptr>& params);

};
