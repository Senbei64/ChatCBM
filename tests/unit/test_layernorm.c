/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/layernorm.s: residuals of the model and random ones, with the
   parameters of several LayerNorms, and a constant vector (zero deviation). */

#include <string.h>
#include "check.h"
#include "vec_layernorm.h"

extern int acc[256], unit[32], x[32];                 /* buffers.s */
extern const unsigned char *ln_g;                     /* layernorm.s */
extern const int *ln_b;
extern int ln_kg;
void layernorm(void);

int main(void)
{
    unsigned c, i;
    for (c = 0; c < CASES; ++c) {
        memcpy(x, in_x + 32 * c, 64);
        ln_g = in_g + 32 * c;
        ln_b = in_b + 32 * c;
        ln_kg = in_kg[c];
        layernorm();
        for (i = 0; i < 32; ++i) {
            check("acc", 32 * c + i, acc[i], out_acc[32 * c + i]);
            check("unit", 32 * c + i, unit[i], out_unit[32 * c + i]);
        }
    }
    return report("layernorm");
}

/*--------------------------------------------------------------------------*/
