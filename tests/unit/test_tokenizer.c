/*- ChatCBM - Copyright 2026 Fabio Carignano -------------------------------*/

/* Unit test of src/tokenizer.s: prompts of questions with upper case, extra spaces,
   tabs and newlines, underscores, digits, apostrophes, empty and too long texts. */

#include <string.h>
#include "check.h"
#include "vec_tokenizer.h"

extern unsigned char prompt_ids[64], prompt_len;      /* tokenizer.s */
unsigned char __fastcall__ tokenize(const char *text);

int main(void)
{
    unsigned c, i;
    const char *text = in_text;
    for (c = 0; c < CASES; ++c) {
        if (tokenize(text)) {
            check("too long", c, 0, out_len[c]);
        } else {
            check("length", c, prompt_len, out_len[c]);
            for (i = 0; i < prompt_len && i < 64; ++i)
                check("id", 64 * c + i, prompt_ids[i], out_ids[64 * c + i]);
        }
        text += strlen(text) + 1;
    }
    return report("tokenizer");
}

/*--------------------------------------------------------------------------*/
