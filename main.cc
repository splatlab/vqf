/*
 * ============================================================================
 *
 *       Filename:  main.cc
 *
 *         Author:  Prashant Pandey (), ppandey2@cs.cmu.edu
 *   Organization:  Carnegie Mellon University
 *
 * ============================================================================
 */
#include <iostream>

#include <stdint.h>
#include <immintrin.h>  // portable to all x86 compilers
#include <tmmintrin.h>

#define SIZE 32

int main()
{
	uint8_t source[SIZE];
	uint8_t order[SIZE];
	for (uint8_t i = 0; i < SIZE; i++) {
		source[i] = i;
		order[i] = i - 1;
	}

	std::cout << "vector before shuffle: \n";
	for (uint8_t i = 0; i < SIZE; i++)
		std::cout << (uint32_t)source[i] << " ";
	std::cout << "\n";

	__m256i vector = _mm256_loadu_si256(reinterpret_cast<__m256i*>(source));
	__m256i shuffle = _mm256_loadu_si256(reinterpret_cast<__m256i*>(order));

	vector = _mm256_shuffle_epi8(vector, shuffle);
	_mm256_storeu_si256(reinterpret_cast<__m256i*>(source), vector);

	std::cout << "vector after shuffle: \n";
	for (uint8_t i = 0; i < SIZE; i++)
		std::cout << (uint32_t)source[i] << " ";
	std::cout << "\n";

	return 0;
}
