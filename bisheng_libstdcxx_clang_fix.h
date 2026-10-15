#pragma once
/*
 * Bisheng (clang 15) + host libstdc++ (GCC 12/13/15) mis-parse constrained
 * std::pair::operator= under C++20 as a redeclaration of the deleted overload
 * in bits/stl_pair.h. Undefine concepts so libstdc++ uses the older path.
 * TileLang ASC still needs -std=c++20 for std::bit_cast in numeric_limits.h.
 */
#if defined(__clang__)
#undef __cpp_concepts
#define __cpp_concepts 0
#endif
