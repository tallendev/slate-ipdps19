#pragma once

#include <string>

#define YY_DECL int yylex(std::string& out, yyscan_t yyscanner, int& brace_counter, bool& simple)

void slate_yylex(std::string& out, void* lexer, bool simple);

void init_scanner(const char* buffer, void** lexer_state);
void kill_scanner(void* lexer_state);
