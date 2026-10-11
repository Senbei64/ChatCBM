/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/linear.s: kernel sums plus the prepared biases, with saturation,
   for 32-output layers and for a block of the logits (output index 64). */

#include <string.h>
#include "check.h"
#include "vec_linear.h"

extern int acc[256], unit[32], kx;                    /* buffers.s */
extern unsigned char acc_lo[32], acc_hi[32];          /* matmul.s */
#pragma zpsym ("acc_lo")
#pragma zpsym ("acc_hi")
extern const unsigned *lin_bias;                      /* linear.s */
extern const signed char *lin_kw;
extern int lin_j0;
extern unsigned char lin_unit;
void lin_post(void);

int main(void)
{
    unsigned c, j, j0;
    for (c = 0; c < CASES; ++c) {
        j0 = in_j0[c];
        for (j = 0; j < 32; ++j) {
            acc_lo[j] = in_sum[32 * c + j] & 255;
            acc_hi[j] = in_sum[32 * c + j] >> 8;
        }
        lin_bias = in_bias + 32 * c - j0;             /* the routine adds j0 itself */
        lin_kw = in_kw + 32 * c - j0;
        lin_j0 = j0;
        lin_unit = j0 == 0;
        kx = in_kx[c];
        lin_post();
        for (j = 0; j < 32; ++j) {
            check("acc", 32 * c + j, acc[j0 + j], out_acc[32 * c + j]);
            if (j0 == 0)
                check("unit", 32 * c + j, unit[j], out_unit[32 * c + j]);
        }
    }
    return report("linear");
}

/*--------------------------------------------------------------------------*/
