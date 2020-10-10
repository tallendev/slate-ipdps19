
%{
// this is where includes and state would go... if i had some
#include "kernel_scan.h"
std::string start_bracket_2d(
    "\n\
    #define SLATE_ITERS 10 \n\
            __shared__ unsigned int id;\n\
    __shared__ uint valid_task;\n\
    uint slate_smid;\n\
    const int leader = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);\n\
    if (leader)\n\
    {\n\
        id = 0;\n\
        slate_smid = get_smid();\n\
        valid_task = slate_smid < sm_low || slate_smid >= sm_high;\n\
    }\n\
    __syncthreads();\n\
    if (valid_task){ return;}\n\
    __shared__ uint3 shared_blockIdxx;\n\
    __shared__ int iters;\n\
    uint local_id;\n\
    do\n\
    {\n\
        if (leader)\n\
        {\n\
            local_id = atomicAdd(&slate_idx, SLATE_ITERS);\n\
            iters = min((int)SLATE_ITERS, (int)slate_max - (int)local_id);\n\
            id = local_id + SLATE_ITERS;\n\
            shared_blockIdxx.x= local_id % gridDimm.x - 1;\n\
            shared_blockIdxx.y = local_id / gridDimm.x;\n\
        }\n\
        __syncthreads();\n\
//        local_id = id - 10;//deletememem\n\
        const int local_iters = iters;\n\
        uint3 blockIdxx;\n\
        blockIdxx.x = shared_blockIdxx.x;\n\
        blockIdxx.y = shared_blockIdxx.y;\n\
        for (int slate_counter = 0; slate_counter < local_iters; ++slate_counter)\n\
        {\n\
            blockIdxx.x++;\n\
            if (blockIdxx.x == gridDimm.x)\n\
            {\n\
                blockIdxx.x = 0;\n\
                blockIdxx.y++;\n\
            }\n\
            //assert (blockIdxx.x < gridDimm.x);\n\
            //assert (blockIdxx.y < gridDimm.y);\n\
        "
         );

std::string start_bracket_simple(
    "\n\
    #define SLATE_ITERS 10 \n\
            __shared__ unsigned int id;\n\
    __shared__ uint valid_task;\n\
    uint slate_smid;\n\
    const int leader = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);\n\
    if (leader)\n\
    {\n\
        id = 0;\n\
        slate_smid = get_smid();\n\
        valid_task = slate_smid < sm_low || slate_smid >= sm_high;\n\
    }\n\
    __syncthreads();\n\
    if (valid_task){ return;}\n\
    __shared__ uint3 shared_blockIdxx;\n\
    __shared__ int iters;\n\
    uint3 blockIdxx;\n\
    uint local_id;\n\
    do\n\
    {\n\
        if (leader)\n\
        {\n\
            local_id = atomicAdd(&slate_idx, SLATE_ITERS);\n\
            iters = min((int)SLATE_ITERS, (int)slate_max - (int)local_id);\n\
            id = local_id + iters;\n\
            shared_blockIdxx.x = local_id;\n\
        }\n\
        __syncthreads();\n\
        blockIdxx.x = shared_blockIdxx.x;\n\
        const register int local_iters = iters;\n\
        for (int slate_counter = 0; slate_counter < local_iters; ++slate_counter, ++blockIdxx.x)\n\
        {\n"
         );




std::string end_bracket("\
        }\n\
    } while(!retreat && id < slate_max);\n");
//BEGIN(INITIAL);
%}

%option reentrant
%option noyywrap

ws   [ \t\f]

%x GLOBAL
%x GLOBAL_END
%x COMMA
%%

__global__ {
               BEGIN(GLOBAL);
               out += std::string(yytext);
           }

<GLOBAL>\( {
              out += std::string(yytext) + "const uint sm_low, const uint sm_high";
              BEGIN(COMMA);
           }

<COMMA>[^{ws}\)] {
                     BEGIN(GLOBAL);
                     out += ", " + std::string(yytext);
                 }

<COMMA>{ws}*\) {
                   BEGIN(GLOBAL);
               }

<GLOBAL>\{ {
                out += std::string(yytext) + (simple ? start_bracket_simple : start_bracket_2d);
                brace_counter++;
                BEGIN(GLOBAL_END);
           }

<GLOBAL_END>blockIdx {
                        out += "blockIdxx";
                }

<GLOBAL_END>gridDim {
                        out += "gridDimm";
                }

<GLOBAL_END>\{ {
                    brace_counter++;
                    out += std::string(yytext);
                }

<GLOBAL_END>\} {
                    brace_counter--;
                    if (brace_counter == 0)
                    {
                        BEGIN(INITIAL);
                        out += end_bracket; 
                    }
                    out += std::string(yytext);
               }
           

<*>"\n" {
            out += std::string("\n");
        }

<*>. { 
         out += std::string(yytext);
     }


 /*
<GLOBAL>\( {
              out += std::string(yytext) + "uint3 bshift, dim3 gshift";
              BEGIN(COMMA);
           }

<GLOBAL>\{   {
                BEGIN(INITIAL);
                out += std::string(yytext) +
"\n\
    uint3 threadId
    while(0)\n";
             }
<COMMA>[^{ws}\)] {
                     BEGIN(GLOBAL);
                     out += ", " + std::string(yytext);
                 }

<COMMA>{ws}*\) {
                   BEGIN(GLOBAL);
               }
 */

%%
/*



*/

void slate_yylex(std::string& out, void* lexer, bool simple)
{
    int brace_counter = 0;
    //printf("yylex\n");
    yylex(out, lexer, brace_counter, simple);
    //printf("yydone\n");
}

void init_scanner(const char* buffer, void** lexer_state)
{
    //printf("init scanner\n");
    yylex_init(lexer_state);
    //printf("copy string\n");
    YY_BUFFER_STATE bs = yy_scan_string(buffer, *lexer_state);
}

void kill_scanner(yyscan_t lexer_state)
{
    //printf("Kill scanner\n");
    //yy_delete_buffer(YY_BUFFER_STATE);
    yylex_destroy(lexer_state);
}
