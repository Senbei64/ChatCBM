/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/sampler.s: argmax and consecutive samples (temperature 0.6,
   top-p 0.9, xorshift32 seeds) on real and synthetic logits. */

#include <string.h>
#include "check.h"
#include "vec_sampler.h"

extern int acc[256];                                  /* buffers.s */
extern unsigned long rng_state;                       /* sampler.s */
unsigned char argmax(void);
unsigned char sample(void);

int main(void)
{
    unsigned c, d;
    for (c = 0; c < CASES; ++c) {
        memcpy(acc, in_z + 256 * c, 512);
        check("argmax", c, argmax(), out_argmax[c]);
        rng_state = in_seed[c];
        for (d = 0; d < DRAWS; ++d)
            check("sample", DRAWS * c + d, sample(), out_draws[DRAWS * c + d]);
    }
    return report("sampler");
}

/*--------------------------------------------------------------------------*/
