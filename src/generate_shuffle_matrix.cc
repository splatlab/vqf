/*
 * ============================================================================
 *
 *       Filename:  generate_shuffle_matrix.cc
 *
 *         Author:  Prashant Pandey (), ppandey@berkeley.edu
 *   Organization:  LBNL/UCB
 *
 * ============================================================================
 */


#include	<stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

/*
#define SHUFFLE_SIZE 32

void generate_shuffle_256(void) {
std::ofstream shuffle_matrix("src/shuffle_matrix_256.c");

// generate right shuffle
for (uint64_t index = 0; index < SHUFFLE_SIZE; index++) {
shuffle_matrix << "const __m256i RM" << std::to_string(index) << " = _mm256_setr_epi8(\n";
for (uint8_t i = 0, j = 0; i < SHUFFLE_SIZE; i++) {
if (i == index) {
shuffle_matrix << std::to_string(SHUFFLE_SIZE - 1);
} else {
shuffle_matrix << std::to_string(j++);
}
if (i < SHUFFLE_SIZE - 1)
shuffle_matrix << ", ";
}
shuffle_matrix << ");\n";
}
shuffle_matrix << '\n';
shuffle_matrix << "const __m256i RM [] = {";
for (uint8_t i = 0; i < SHUFFLE_SIZE; i++) {
shuffle_matrix << "RM" << std::to_string(0) << ", ";
}
for (uint8_t i = 0; i < SHUFFLE_SIZE; i++) {
shuffle_matrix << "RM" << std::to_string(i);
if (i < SHUFFLE_SIZE - 1)
shuffle_matrix << ", ";
}
shuffle_matrix << "};\n";

shuffle_matrix << "\n\n\n";

// generate left shuffle
for (uint64_t index = 0; index < SHUFFLE_SIZE; index++) {
shuffle_matrix << "const __m256i LM" << std::to_string(index) << " = _mm256_setr_epi8(\n";
for (uint8_t i = 0, j = 0; i < SHUFFLE_SIZE; i++) {
if (i == index) {
shuffle_matrix << std::to_string(SHUFFLE_SIZE - 1);
} else {
shuffle_matrix << std::to_string(j++);
}
if (i < SHUFFLE_SIZE - 1)
shuffle_matrix << ", ";
}
shuffle_matrix << ");\n";
}
shuffle_matrix << '\n';
shuffle_matrix << "const __m256i LM [] = {";
for (uint8_t i = 0; i < SHUFFLE_SIZE; i++) {
shuffle_matrix << "LM" << std::to_string(i) << ", ";
}
for (uint8_t i = 0; i < SHUFFLE_SIZE; i++) {
shuffle_matrix << "LM" << std::to_string(31);
if (i < SHUFFLE_SIZE - 1)
shuffle_matrix << ", ";
}

shuffle_matrix << "};\n";
}

*/

#define SHUFFLE_SIZE 64

void generate_shuffle_512(void) {
   std::ofstream shuffle_matrix("src/shuffle_matrix_512.c");

   shuffle_matrix << "#include <immintrin.h>\n#include <tmmintrin.h>\n\n";
   // generate right shuffle
   for (uint64_t index = 0; index < SHUFFLE_SIZE; index++) {
      shuffle_matrix << "const __m512i S" << std::to_string(index) << " = _mm512_set_epi8(\n";
      for (uint8_t i = 0, j = SHUFFLE_SIZE - 2; i < SHUFFLE_SIZE; i++) {
         if (i == SHUFFLE_SIZE - index - 1) {
            shuffle_matrix << std::to_string(SHUFFLE_SIZE - 1);
         } else {
            shuffle_matrix << std::to_string(j--);
         }
         if (i < SHUFFLE_SIZE - 1)
            shuffle_matrix << ", ";
      }
      shuffle_matrix << ");\n";
   }
   shuffle_matrix << '\n';
   shuffle_matrix << "const __m512i SHUFFLE [] = {";
   for (uint8_t i = 0; i < SHUFFLE_SIZE; i++) {
      shuffle_matrix << "S" << std::to_string(i);
      if (i < SHUFFLE_SIZE - 1)
         shuffle_matrix << ", ";
   }
   shuffle_matrix << "};\n";

   shuffle_matrix << "\n";
   // generate left shift
   for (uint64_t index = 0; index < SHUFFLE_SIZE; index++) {
      shuffle_matrix << "const __m512i R" << std::to_string(index) << " = _mm512_set_epi8(\n";
      shuffle_matrix << std::to_string(SHUFFLE_SIZE - 1) << ", "; // always overwrite the last item
      for (uint8_t i = 0, j = SHUFFLE_SIZE - 1; i < SHUFFLE_SIZE - 1; i++) {
         if (i == SHUFFLE_SIZE - index - 1) {
            j--;
            shuffle_matrix << std::to_string(j--);
         } else {
            shuffle_matrix << std::to_string(j--);
         }
         if (i < SHUFFLE_SIZE - 2)
            shuffle_matrix << ", ";
      }
      shuffle_matrix << ");\n";
   }
   shuffle_matrix << '\n';
   shuffle_matrix << "const __m512i SHUFFLE_REMOVE [] = {";
   for (uint8_t i = 0; i < SHUFFLE_SIZE; i++) {
      shuffle_matrix << "R" << std::to_string(i);
      if (i < SHUFFLE_SIZE - 1)
         shuffle_matrix << ", ";
   }
   shuffle_matrix << "};\n";
}

#define SHUFFLE_SIZE 32
void generate_shuffle_512_16(void) {
   std::ofstream shuffle_matrix("src/shuffle_matrix_512_16.c");

   shuffle_matrix << "#include <immintrin.h>\n#include <tmmintrin.h>\n\n";
   // generate right shuffle
   for (uint64_t index = 0; index < SHUFFLE_SIZE; index++) {
      shuffle_matrix << "const __m512i S16_" << std::to_string(index) << " = _mm512_set_epi16(\n";
      for (uint8_t i = 0, j = SHUFFLE_SIZE - 2; i < SHUFFLE_SIZE; i++) {
         if (i == SHUFFLE_SIZE - index - 1) {
            shuffle_matrix << std::to_string(SHUFFLE_SIZE - 1);
         } else {
            shuffle_matrix << std::to_string(j--);
         }
         if (i < SHUFFLE_SIZE - 1)
            shuffle_matrix << ", ";
      }

      shuffle_matrix << ");\n";
   }
   shuffle_matrix << '\n';
   shuffle_matrix << "__m512i SHUFFLE16 [] = {";
   for (uint8_t i = 0; i < SHUFFLE_SIZE; i++) {
      shuffle_matrix << "S16_" << std::to_string(i);
      if (i < SHUFFLE_SIZE - 1)
         shuffle_matrix << ", ";
   }
   shuffle_matrix << "};\n";

   shuffle_matrix << "\n";
   // generate left shift
   for (uint64_t index = 0; index < SHUFFLE_SIZE; index++) {
      shuffle_matrix << "const __m512i R16_" << std::to_string(index) << " = _mm512_set_epi16(\n";
      shuffle_matrix << std::to_string(SHUFFLE_SIZE - 1) << ", "; // always overwrite the last item
      for (uint8_t i = 0, j = SHUFFLE_SIZE - 1; i < SHUFFLE_SIZE - 1; i++) {
         if (i == SHUFFLE_SIZE - index - 1) {
            j--;
            shuffle_matrix << std::to_string(j--);
         } else {
            shuffle_matrix << std::to_string(j--);
         }
         if (i < SHUFFLE_SIZE - 2)
            shuffle_matrix << ", ";
      }

      shuffle_matrix << ");\n";
   }
   shuffle_matrix << '\n';
   shuffle_matrix << "__m512i SHUFFLE_REMOVE16 [] = {";
   for (uint8_t i = 0; i < SHUFFLE_SIZE; i++) {
      shuffle_matrix << "R16_" << std::to_string(i);
      if (i < SHUFFLE_SIZE - 1)
         shuffle_matrix << ", ";
   }
   shuffle_matrix << "};\n";
}

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  main
 *  Description:  
 * ============================================================================
 */
   int
main ( int argc, char *argv[] )
{
   generate_shuffle_512_16();
   return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
