#pragma once

enum Command : unsigned int { 
                                cumemset      = 499u,
                                cumalloc      = 500u,
                                cpytodevice   = 501u,
                                cpyfromdevice = 502u,
                                cukernel        = 503u,
                                cusync        = 504u,
                                hangup        = 505u
                            };

enum SlateStatus : unsigned int { 
                                    sync_success = 251,
                                    read_done = 296,
                                    err = 999 
                                };
