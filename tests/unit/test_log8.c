/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/log8.s: to_log8 (one scale per vector) and enc_static (a fixed
   scale per head, written with a stride as in the KV cache). */

#include <string.h>
#include "check.h"
#include "vec_log8.h"

#define STRIDE 3

extern int acc[256], unit[32], kx;                    /* buffers.s */
extern unsigned char xs[32], xe[32];
extern unsigned char *enc_out;                        /* log8.s */
extern const int *enc_k;
extern unsigned char enc_stride;
void to_log8(void);
void enc_static(void);

static unsigned char out[32 * STRIDE];

int main(void)
{
    unsigned c, i;
    for (c = 0; c < CASES; ++c) {
        memcpy(acc, in_acc + 32 * c, 64);
        memcpy(unit, in_unit + 32 * c, 64);
        to_log8();
        for (i = 0; i < 32; ++i) {
            check("xs", 32 * c + i, xs[i], out_s[32 * c + i]);
            check("xe", 32 * c + i, xe[i], out_e[32 * c + i]);
        }
        check("kx", c, kx, out_k[c]);

        enc_out = out;
        enc_k = in_K + 4 * c;
        enc_stride = STRIDE;
        enc_static();
        for (i = 0; i < 32; ++i)
            check("enc", 32 * c + i, out[STRIDE * i], out_enc[32 * c + i]);
    }
    return report("log8");
}

/*--------------------------------------------------------------------------*/
