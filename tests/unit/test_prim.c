/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/prim.s: ilog16u and pow2_16 against the Python engine. */

#include "check.h"
#include "vec_prim.h"

int __fastcall__ ilog16u(unsigned v);
long __fastcall__ pow2_16(int e);

int main(void)
{
    unsigned i;
    for (i = 0; i < ILOG_N; ++i)
        check("ilog16u", ilog_in[i], ilog16u(ilog_in[i]), ilog_out[i]);
    for (i = 0; i < POW_N; ++i)
        check("pow2_16", i, pow2_16(pow_in[i]), pow_out[i]);
    return report("prim");
}

/*--------------------------------------------------------------------------*/
