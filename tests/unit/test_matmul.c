/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/matmul.s: 32 x 32 blocks of real packed nf4 weights, with real
   inputs (LayerNorm outputs of the model) and random ones with zero elements. */

#include <string.h>
#include "check.h"
#include "vec_matmul.h"

extern unsigned char xs[32], xe[32];                  /* buffers.s */
extern unsigned char prodtab[256], lutlo[256], luthi[256];
extern unsigned char acc_lo[32], acc_hi[32];
#pragma zpsym ("acc_lo")
#pragma zpsym ("acc_hi")
extern unsigned char wbuf[512];                       /* aligned.s */
void __fastcall__ matmul(const unsigned char *w);

int main(void)
{
    unsigned c, j;
    memcpy(prodtab, prod, 256);
    for (j = 0; j < 256; ++j) {
        lutlo[j] = nf4log[j & 15];
        luthi[j] = nf4log[j >> 4];
    }
    for (c = 0; c < CASES; ++c) {
        memcpy(wbuf, weights + 512 * c, 512);
        memcpy(xs, in_s + 32 * c, 32);
        memcpy(xe, in_e + 32 * c, 32);
        matmul(wbuf);
        for (j = 0; j < 32; ++j)
            check("acc", 32 * c + j, (int)(acc_lo[j] | (acc_hi[j] << 8)), out_acc[32 * c + j]);
    }
    return report("matmul");
}

/*--------------------------------------------------------------------------*/
