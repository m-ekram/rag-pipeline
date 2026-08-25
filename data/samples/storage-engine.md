# Halyard Storage Engine - Operator Guide

## Overview

Halyard is a log-structured merge-tree store. Writes land in an in-memory
memtable, are flushed to immutable SSTables on disk, and are merged by a
background compactor. Reads consult the memtable first, then SSTables newest to
oldest, using a bloom filter per table to skip files that cannot contain the key.

## Configuration keys

All keys live under the `[storage]` section of `halyard.toml`.

- `memtable_size_mb` (default 64) - flush threshold. Larger values reduce write
  amplification but lengthen recovery after a crash.
- `max_open_files` (default 1024) - raise this on hosts with many SSTables;
  running out surfaces as `HAL_E_TOO_MANY_FILES`.
- `compaction_threads` (default 4) - background compactor parallelism. Set to
  the number of physical cores minus two on write-heavy nodes.
- `bloom_bits_per_key` (default 10) - 10 bits gives roughly a 1% false positive
  rate. Raising it to 16 cuts false positives to about 0.05% at the cost of
  memory.
- `block_cache_mb` (default 512) - the single most effective read-latency knob.
- `wal_sync` (default `batch`) - one of `never`, `batch`, `always`. `always` is
  the only setting that survives sudden power loss without data loss.
- `compression` (default `lz4`) - `none`, `lz4`, or `zstd`. `zstd` gives about
  30% better ratio at roughly double the CPU cost.

## Compaction

Halyard uses levelled compaction. Level 0 holds freshly flushed tables that may
overlap; every level below it holds non-overlapping tables and is ten times the
size of the level above.

Write stalls happen when level 0 accumulates more than
`level0_stall_trigger` (default 20) files - the engine deliberately throttles
writes so the compactor can catch up. Persistent stalls mean the compactor is
under-provisioned: raise `compaction_threads` or move to faster disks.

## Backups

Halyard supports online snapshots via hard links, so a snapshot costs almost no
space at the moment it is taken:

    halyardctl snapshot create --name nightly-$(date +%F)

Snapshots are consistent as of the moment they are taken but they are **not**
backups until copied off the host. Restore with:

    halyardctl snapshot restore --name nightly-2024-05-01 --target /var/lib/halyard

Restore requires the engine to be stopped. Expect roughly 4 minutes per 100 GB
on NVMe.

## Common errors

| Error                  | Cause                                            |
|------------------------|--------------------------------------------------|
| HAL_E_TOO_MANY_FILES   | `max_open_files` below what the level count needs |
| HAL_E_WAL_CORRUPT      | Torn write after power loss with `wal_sync=batch` |
| HAL_E_COMPACTION_STALL | Level 0 above the stall trigger for over a minute |
| HAL_E_CHECKSUM         | Block checksum mismatch - usually failing disk    |

## Tuning for read-heavy workloads

Raise `block_cache_mb` to about 40% of host RAM, raise `bloom_bits_per_key` to
16, and leave compression at `lz4`. Do not raise `memtable_size_mb` for read
workloads - it does not help reads and it lengthens crash recovery.
