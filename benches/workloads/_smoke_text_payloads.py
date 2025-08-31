from benches.workloads.text_payloads import (
    long_sentences_factory,
    short_sentence_factory,
    utf8_sizeof,
)

short = short_sentence_factory()
longf = long_sentences_factory()

for k in ["S:k0", "S:k1", "S:k2"]:
    s = short(k)
    print("SHORT:", k, "|", s, "| bytes:", utf8_sizeof(s))

for k in ["N:k99998", "N:k99999"]:
    t = longf(k)
    print("LONG :", k, "|", t[:80] + "...", "| bytes:", utf8_sizeof(t))
