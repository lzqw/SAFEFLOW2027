# Measured-memory serial admission, November14 11:30

Both EA arms genuinely completed100K new transitions: control4101.91s,
repair4151.31s; completed receipts retained. The old queue snapshot still
called a finished zombie child live until its next1800s poll. Do not infer
training liveness from that cached label. No EA retry or continuation is made.

4080 now has only3996MiB free while the foreign project's11488MiB allocation
remains; observed GPU utilization2%. A100's two GPUs remain100% busy, with
5408/5448MiB free. Do not interrupt those projects, and do not use5080.
The initially conservative6GiB admission rule would strand our small learners
even while4080 compute is temporarily idle. This is a RESOURCE amendment,
not a changed scientific risk or collision acceptance rule.

One bounded disposable memory probe used the actual frozen native Iso120K
and SAC300K parent states, batch256, four update rounds including two actor
updates each. It collected ZERO environment transitions, saved no learned
model, changed no scientific checkpoint and discarded all probe updates.
Iso peak Torch allocated/reserved24,788,480/29,360,128bytes, observed native
process332MiB; SAC21,912,064/27,262,976bytes, process330MiB. These measurements
do not bound all future peaks or foreign memory growth.

Before admitting the remaining TWO original100K jobs, explicitly retire the
idle old queue and still-waiting validation controller after checking their
exact commands, both EA receipts and absence of live training children.
Preserve their files and source directories. New immutable source directory,
new controller state, new validation output; no duplicate trainer or hot edit.

The replacement admits at most ONE own GPU trainer at a time, only with
at least3072MiB free and40GiB disk headroom, and limits its own Torch CUDA
allocator to12.5% of the16GiB GPU (about2GiB; context/non-Torch allocations
are additional). This is much larger than measured learner allocations,
but does NOT guarantee against OOM. Engineering failure still stops without
automatic retry; preserve and count every actual transition. Never reduce
batch size, updates, environment count or any scientific budget to fit memory.

Only admission resources change: seed41028, original parents, all RNG/state,
reset family, reward, safety tests, optimizer, dual, batch256, two environments,
100K new transitions each and five saved candidates remain unchanged.
The unchanged4000-episode common validation gets a replacement wait-controller
bound to the new queue; final91027–91126 labels remain unopened.
E01 spending remains200000 completed +200000 pending +325 failed prefix,
not a new training reservation. CPU memory probing updates are disclosed as
disposable compute, not scientific transitions or new trained policies.
