# 4080 connection recovery — November14,2026

Read-only observation at09:26:28+08:00: host447, usercarla, RTX4080 SUPER16376MiB,
698MiB observed GPU memory used. IPv6 SSH succeeded with strict host-key checking
against the previously recorded IPv4 identity. The local `4080` SSH alias now
uses IPv6; `4080-ipv4` preserves the old route. Credentials/private keys are not
included in this report.

`/home/carla/4T` is a mounted ext4 data filesystem (`/dev/sda1`),3.6TiB total,
approximately865GiB available. Original Panda Payload full300K checkpoint files:

| Run under /home/carla/4T/SAFEFLOW2027_PAPER/runs | File | Bytes |
|---|---|---:|
|payload-sac-41027-v1|full_300000.pt|172185885|
|payload-iso-41027-v1|full_300000.pt|172587397|
|payload-ea-41027-v1|full_300000.pt|172570629|
|payload-sac-41028-v1|full_300000.pt|172165981|
|payload-iso-41028-v1|full_300000.pt|172495301|
|payload-ea-41028-v1|full_300000.pt|172527749|

This confirms file presence and sizes, not full schema/parent compatibility,
checkpoint integrity or successful resume. No checkpoint is reconstructed,
copied into Git, loaded, or used to start repair in this connection/export task.
No foreign job is changed. The previously recorded no-route blocker is resolved;
scientific repair remains unstarted. Snapshot GPU/disk use is not a reservation.
