# Engineering failure, retained rather than a method result

Source86669ee, three requested profiles32768transitions each. All three exited
at their first preheat logging call because `metrics()` stacked an empty list
before any learner update. No scientific candidate was trained and no throughput
ranking was obtained. The parent finite controller4067293 was explicitly stopped
after diagnosis; all three native children are terminal, and none was automatically
restarted. V2 uses new output paths and an explicit source revision.

The committed code's first-log threshold is `10000//E*E`; initialization starts
at0, advances exactlyE per native step, and the stacktrace occurs after the
cost-to-CPU synchronization. Thus the collected prefixes inferred from actual
exception location plus executed source/configuration are9984(Balanced),
9728(Fast),9984(Learn-heavy), total29696. These are **source-inferred**, not
pretended native receipt counters. There are no persisted native step/cost
snapshots for these failed prefixes. Training costs are unknown, not zero.

For budget safety, reserve the entire three requested prefixes98304 rather
than use an unverified zero charge. Original logs/configurations/header-only
progress files are retained. The repair handles empty warmup metrics and adds
an exit-time full-state/prefix checkpoint for future exceptions. Native dynamics,
reward/cost, environment batch, update schedules and the three configurations do
not change. The one CPU empty-metrics regression check passed, with0physics.
