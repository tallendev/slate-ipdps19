#pragma once
#include <exception>

class nvrtc_exception: public std::exception                                     
{                                                                               
    public:
            virtual const char* what() const throw()                                
            {                                                                       
                return "NVRTCException: Compiler/Linker error"; 
            }                                                                       
};       
