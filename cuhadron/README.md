# A synchronization category for cuHadron

[cuHadron](https://github.com/NVlabs/cuHadron) has no tests of synchronization between thread blocks through global
memory. `scopedsync/` adds six, each with a racy build and a `FIXED` build: relaxed-flag message passing, a volatile
flag missing the consumer fence, a volatile flag missing the producer fence, a relaxed atomic-counter handoff, a
last-block reduction without its fence, and a spinlock without fences. `add_category.sh /path/to/cuHadron` installs
the category in a checkout.

On an H100, syncsan detects all six racy builds and none of the fixed ones; compute-sanitizer racecheck, which checks
shared memory only, detects none.
