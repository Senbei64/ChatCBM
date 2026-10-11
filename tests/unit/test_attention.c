/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/attention.s: heads of the first layer over a real prompt, and
   random caches up to the full 64-token context with left, right and no shift. */

#include <string.h>
#include "check.h"
#include "vec_attention.h"

extern unsigned char att_q[8], att_n, prodatab[256];  /* attention.s */
extern signed char att_zs;
extern const unsigned char *att_kb, *att_vb;
extern int att_y[8];
void att_head(void);

static unsigned char vrows[64 * 32];                  /* rows of 32 bytes, 8 used */

int main(void)
{
    unsigned c, t, d;
    memcpy(prodatab, prod_attn, 256);
    for (c = 0; c < CASES; ++c) {
        memcpy(att_q, in_q + 8 * c, 8);
        for (t = 0; t < 64; ++t)
            memcpy(vrows + 32 * t, in_v + 512 * c + 8 * t, 8);
        att_kb = in_k + 512 * c;                      /* 8 rows of 64 positions */
        att_vb = vrows;
        att_n = in_n[c];
        att_zs = in_zs[c];
        att_head();
        for (d = 0; d < 8; ++d)
            check("y", 8 * c + d, att_y[d], out_y[8 * c + d]);
    }
    return report("attention");
}

/*--------------------------------------------------------------------------*/
